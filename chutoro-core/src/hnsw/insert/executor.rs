//! Applies staged HNSW insertions by mutating the graph and scheduling trim
//! jobs.

use std::collections::{HashMap, HashSet};

use crate::hnsw::{
    error::HnswError,
    graph::{ApplyContext, EdgeContext, Graph, NodeContext},
    types::InsertionPlan,
};

/// Captures the neighbour candidates for a node that may require trimming.
///
/// Each candidate has a corresponding insertion sequence used to implement the
/// deterministic tie-break when trimming applies.
///
/// # Examples
/// ```rust,ignore
/// use crate::hnsw::insert::executor::{EdgeContext, TrimJob};
///
/// let ctx = EdgeContext { level: 0, max_connections: 2 };
/// let job = TrimJob {
///     node: 1,
///     ctx,
///     candidates: vec![2, 3],
///     sequences: vec![4, 5],
/// };
/// assert_eq!(job.candidates.len(), job.sequences.len());
/// ```
#[derive(Clone, Debug)]
pub(crate) struct TrimJob {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
    pub(crate) sequences: Vec<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedInsertion {
    pub(crate) node: NodeContext,
    pub(crate) promote_entry: bool,
    pub(crate) new_node_neighbours: Vec<Vec<usize>>,
    pub(crate) updates: Vec<StagedUpdate>,
    pub(crate) max_connections: usize,
}

/// Captures the staged neighbour set for a node at a given level.
#[derive(Clone, Debug)]
pub(crate) struct StagedUpdate {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
}

/// Stores the final trimmed neighbour list for a node and level.
#[derive(Clone, Debug)]
pub(crate) struct TrimResult {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) neighbours: Vec<usize>,
}

type FinalisedUpdate = (StagedUpdate, Vec<usize>);

/// Captures the accumulated state produced while staging insertion layers.
type LayerProcessingOutcome = (
    Vec<Vec<usize>>,
    HashMap<(usize, usize), Vec<usize>>,
    HashSet<(usize, usize)>,
    HashSet<(usize, usize)>,
);

struct TrimWork {
    staged: HashMap<(usize, usize), Vec<usize>>,
    needs_trim: HashSet<(usize, usize)>,
    max_connections: usize,
}

#[derive(Debug)]
pub(crate) struct InsertionExecutor<'graph> {
    graph: &'graph mut Graph,
}

impl<'graph> InsertionExecutor<'graph> {
    pub(crate) fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    /// Prepares an insertion commit by staging all neighbour list updates.
    ///
    /// The returned [`PreparedInsertion`] captures the new node metadata and
    /// the neighbour lists for all affected nodes as they would appear after
    /// linking. Trimming is deferred to the caller, which should compute
    /// distances without holding the graph lock and then call
    /// [`InsertionExecutor::commit`] with the resulting [`TrimResult`]s.
    pub(crate) fn apply(
        &mut self,
        node: NodeContext,
        apply_ctx: ApplyContext<'_>,
    ) -> Result<(PreparedInsertion, Vec<TrimJob>), HnswError> {
        let ApplyContext { params, plan } = apply_ctx;
        let NodeContext {
            node,
            level,
            sequence,
        } = node;
        self.ensure_slot_available(node)?;
        let promote_entry = level > self.graph.entry().map(|entry| entry.level).unwrap_or(0);
        let max_connections = params.max_connections();
        let (mut new_node_neighbours, staged, _initialised, needs_trim) = self
            .process_insertion_layers(
                NodeContext {
                    node,
                    level,
                    sequence,
                },
                plan,
                max_connections,
            )?;
        Self::dedupe_new_node_lists(&mut new_node_neighbours);
        let (updates, trim_jobs) = self.generate_updates_and_trim_jobs(
            NodeContext {
                node,
                level,
                sequence,
            },
            TrimWork {
                staged,
                needs_trim,
                max_connections,
            },
        )?;

        Ok((
            PreparedInsertion {
                node: NodeContext {
                    node,
                    level,
                    sequence,
                },
                promote_entry,
                new_node_neighbours,
                updates,
                max_connections,
            },
            trim_jobs,
        ))
    }

    fn ensure_slot_available(&self, node: usize) -> Result<(), HnswError> {
        if !self.graph.has_slot(node) {
            return Err(HnswError::InvalidParameters {
                reason: format!("node {node} is outside pre-allocated capacity"),
            });
        }
        if self.graph.node(node).is_some() {
            return Err(HnswError::DuplicateNode { node });
        }
        Ok(())
    }

    /// Processes the insertion layers, staging neighbour lists and identifying
    /// nodes that will require trimming once distances are available.
    ///
    /// The provided [`NodeContext`] identifies the new node and the highest
    /// level that should be considered during staging.
    fn process_insertion_layers(
        &self,
        ctx: NodeContext,
        plan: InsertionPlan,
        max_connections: usize,
    ) -> Result<LayerProcessingOutcome, HnswError> {
        let mut new_node_neighbours = vec![Vec::new(); ctx.level + 1];
        let mut staged: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut initialised = HashSet::new();
        let mut needs_trim = HashSet::new();

        for layer in plan
            .layers
            .into_iter()
            .filter(|layer| layer.level <= ctx.level)
        {
            let level_index = layer.level;

            for neighbour in layer.neighbours.into_iter().take(max_connections) {
                self.stage_neighbour(
                    ctx.node,
                    neighbour.id,
                    level_index,
                    max_connections,
                    &mut new_node_neighbours,
                    &mut staged,
                    &mut initialised,
                    &mut needs_trim,
                )?;
            }
        }

        Ok((new_node_neighbours, staged, initialised, needs_trim))
    }

    /// Builds staged updates and trimming jobs from the collected neighbour
    /// candidates.
    fn generate_updates_and_trim_jobs(
        &self,
        new_node: NodeContext,
        work: TrimWork,
    ) -> Result<(Vec<StagedUpdate>, Vec<TrimJob>), HnswError> {
        let TrimWork {
            mut staged,
            needs_trim,
            max_connections,
        } = work;
        let mut updates = Vec::with_capacity(staged.len());
        let mut trim_jobs = Vec::with_capacity(needs_trim.len());

        for ((other, lvl), mut candidates) in staged.drain() {
            Self::dedupe_candidates(&mut candidates);
            let ctx = EdgeContext {
                level: lvl,
                max_connections,
            };
            prioritise_new_node(new_node.node, &mut candidates);
            let mut sequences = Vec::with_capacity(candidates.len());
            for &candidate in &candidates {
                sequences.push(self.sequence_for_candidate(candidate, new_node, lvl)?);
            }
            if needs_trim.contains(&(other, lvl)) {
                let reordered = candidates.clone();
                debug_assert_eq!(
                    reordered.len(),
                    sequences.len(),
                    "trim job sequences must align with candidates",
                );
                trim_jobs.push(TrimJob {
                    node: other,
                    ctx,
                    candidates: reordered,
                    sequences,
                });
            }
            updates.push(StagedUpdate {
                node: other,
                ctx,
                candidates,
            });
        }

        Ok((updates, trim_jobs))
    }

    fn dedupe_new_node_lists(levels: &mut [Vec<usize>]) {
        for neighbours in levels {
            let mut seen = HashSet::new();
            neighbours.retain(|neighbour| seen.insert(*neighbour));
        }
    }

    fn dedupe_candidates(candidates: &mut Vec<usize>) {
        candidates.sort_unstable();
        candidates.dedup();
    }

    fn sequence_for_candidate(
        &self,
        candidate: usize,
        new_node: NodeContext,
        level: usize,
    ) -> Result<u64, HnswError> {
        if candidate == new_node.node {
            return Ok(new_node.sequence);
        }
        self.graph
            .node_sequence(candidate)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!(
                    "insertion planning: sequence missing for node {candidate} at level {level}"
                ),
            })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Staging shares tightly-coupled accumulators; refactoring into a tracker is follow-up work"
    )]
    fn stage_neighbour(
        &self,
        new_node: usize,
        neighbour: usize,
        level_index: usize,
        max_connections: usize,
        new_node_neighbours: &mut [Vec<usize>],
        staged: &mut HashMap<(usize, usize), Vec<usize>>,
        initialised: &mut HashSet<(usize, usize)>,
        needs_trim: &mut HashSet<(usize, usize)>,
    ) -> Result<(), HnswError> {
        new_node_neighbours[level_index].push(neighbour);

        let key = (neighbour, level_index);
        if initialised.insert(key) {
            let graph_node =
                self.graph
                    .node(neighbour)
                    .ok_or_else(|| HnswError::GraphInvariantViolation {
                        message: format!(
                            "insertion planning: node {neighbour} missing at level {level_index}",
                        ),
                    })?;
            staged.insert(key, graph_node.neighbours(level_index).to_vec());
        }

        let candidates =
            staged
                .get_mut(&key)
                .ok_or_else(|| HnswError::GraphInvariantViolation {
                    message: format!(
                        "insertion planning: node {neighbour} missing from staged updates at level {level_index}",
                    ),
                })?;

        let contains_new = candidates.contains(&new_node);
        let projected = candidates.len() + usize::from(!contains_new);
        if projected > max_connections {
            needs_trim.insert(key);
        }
        if contains_new {
            return Ok(());
        }

        candidates.push(new_node);
        Ok(())
    }

    /// Applies a prepared insertion after trim distances have been evaluated.
    pub(crate) fn commit(
        &mut self,
        prepared: PreparedInsertion,
        trims: Vec<TrimResult>,
    ) -> Result<(), HnswError> {
        let PreparedInsertion {
            node,
            promote_entry,
            new_node_neighbours,
            updates,
            max_connections,
        } = prepared;

        let mut trim_lookup: HashMap<(usize, usize), Vec<usize>> = trims
            .into_iter()
            .map(|result| ((result.node, result.ctx.level), result.neighbours))
            .collect();

        let original_new_node_neighbours = new_node_neighbours.clone();
        let (final_updates, mut reciprocity) =
            Self::build_reciprocity_map(updates, &mut trim_lookup, node.node);
        let filtered_new_node_neighbours = Self::filter_neighbours_by_reciprocity(
            new_node_neighbours,
            &original_new_node_neighbours,
            &mut reciprocity,
            node.level,
        );

        self.graph.attach_node(node)?;
        self.apply_new_node_neighbours(node.node, node.level, filtered_new_node_neighbours);
        if promote_entry {
            self.graph.promote_entry(node.node, node.level);
        }

        self.apply_neighbour_updates(final_updates)?;
        self.enforce_bidirectional(max_connections);
        Ok(())
    }
}

impl<'graph> InsertionExecutor<'graph> {
    /// Builds the final neighbour lists for staged updates while tracking which
    /// nodes reciprocated connections to the newly inserted node per layer.
    fn build_reciprocity_map(
        updates: Vec<StagedUpdate>,
        trim_lookup: &mut HashMap<(usize, usize), Vec<usize>>,
        new_node: usize,
    ) -> (Vec<FinalisedUpdate>, HashMap<usize, HashSet<usize>>) {
        let mut final_updates = Vec::with_capacity(updates.len());
        let mut reciprocity: HashMap<usize, HashSet<usize>> = HashMap::new();

        for update in updates {
            let neighbours = trim_lookup
                .remove(&(update.node, update.ctx.level))
                .unwrap_or_else(|| update.candidates.clone());
            if neighbours.contains(&new_node) {
                reciprocity
                    .entry(update.ctx.level)
                    .or_default()
                    .insert(update.node);
            }
            final_updates.push((update, neighbours));
        }

        (final_updates, reciprocity)
    }

    /// Filters the new node's neighbours to retain only nodes that preserved
    /// reciprocal links after trimming, falling back to the original adjacency
    /// when reciprocity drops every candidate on a layer.
    fn filter_neighbours_by_reciprocity(
        mut new_node_neighbours: Vec<Vec<usize>>,
        original_new_node_neighbours: &[Vec<usize>],
        reciprocity: &mut HashMap<usize, HashSet<usize>>,
        node_level: usize,
    ) -> Vec<Vec<usize>> {
        for (level, neighbours) in new_node_neighbours
            .iter_mut()
            .enumerate()
            .take(node_level + 1)
        {
            if let Some(reciprocated) = reciprocity.get(&level) {
                neighbours.retain(|candidate| reciprocated.contains(candidate));
            }
            let fallback = original_new_node_neighbours
                .get(level)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            Self::ensure_neighbour_connectivity(level, neighbours, reciprocity, fallback);
        }
        new_node_neighbours
    }

    /// Ensures each layer retains at least one neighbour by reinserting the
    /// original candidate used during staging when reciprocity drops the entire
    /// set after trimming.
    fn ensure_neighbour_connectivity(
        level: usize,
        neighbours: &mut Vec<usize>,
        reciprocity: &mut HashMap<usize, HashSet<usize>>,
        fallback: &[usize],
    ) {
        if neighbours.is_empty() {
            if let Some(&fallback_candidate) = fallback.first() {
                neighbours.push(fallback_candidate);
                reciprocity
                    .entry(level)
                    .or_default()
                    .insert(fallback_candidate);
            }
        }
    }

    /// Writes the filtered neighbour lists back to the newly attached node.
    fn apply_new_node_neighbours(
        &mut self,
        node_id: usize,
        node_level: usize,
        filtered_neighbours: Vec<Vec<usize>>,
    ) {
        let node_ref = self
            .graph
            .node_mut(node_id)
            .expect("node was attached above");
        for (level, neighbours) in filtered_neighbours
            .into_iter()
            .enumerate()
            .take(node_level + 1)
        {
            let list = node_ref.neighbours_mut(level);
            list.clear();
            list.extend(neighbours);
        }
    }

    /// Applies the neighbour updates gathered during staging to the existing
    /// nodes now that their adjacency lists have been trimmed.
    fn apply_neighbour_updates(
        &mut self,
        final_updates: Vec<FinalisedUpdate>,
    ) -> Result<(), HnswError> {
        for (update, neighbours) in final_updates {
            let node_ref = self.graph.node_mut(update.node).ok_or_else(|| {
                HnswError::GraphInvariantViolation {
                    message: format!("node {} missing during insertion commit", update.node),
                }
            })?;
            let list = node_ref.neighbours_mut(update.ctx.level);
            list.clear();
            list.extend(neighbours);
        }
        Ok(())
    }

    /// Ensures every edge retains a reciprocal counterpart when trimming evicts
    /// the new link but the source still has capacity. Falls back to removing
    /// the forward edge when reciprocity cannot be restored without exceeding
    /// the layer's degree bounds.
    fn enforce_bidirectional(&mut self, max_connections: usize) {
        let edges = self.collect_all_edges();

        for (origin, target, level) in edges {
            let needs_reverse = self.ensure_reverse_edge(origin, target, level, max_connections);

            if needs_reverse {
                self.remove_forward_edge(origin, target, level);
            }
        }
    }

    /// Collects all edges from the graph as (origin, target, level) tuples.
    fn collect_all_edges(&self) -> Vec<(usize, usize, usize)> {
        let mut edges = Vec::new();
        for (origin, node) in self.graph.nodes_iter() {
            for (level, target) in node.iter_neighbours() {
                edges.push((origin, target, level));
            }
        }
        edges
    }

    /// Computes the connection limit for a given level (doubled for level 0).
    fn compute_connection_limit(&self, level: usize, max_connections: usize) -> usize {
        if level == 0 {
            max_connections.saturating_mul(2)
        } else {
            max_connections
        }
    }

    /// Ensures a reverse edge exists from target to origin. Returns true if
    /// the forward edge should be removed due to capacity constraints.
    #[expect(
        clippy::too_many_arguments,
        reason = "API must surface origin, target, level, and fan-out cap separately"
    )]
    fn ensure_reverse_edge(
        &mut self,
        origin: usize,
        target: usize,
        level: usize,
        max_connections: usize,
    ) -> bool {
        let limit = self.compute_connection_limit(level, max_connections);
        let Some(target_node) = self.graph.node_mut(target) else {
            return false;
        };

        let neighbours = target_node.neighbours_mut(level);

        if neighbours.contains(&origin) {
            return false;
        }

        if neighbours.len() < limit {
            neighbours.push(origin);
            false
        } else {
            true
        }
    }

    /// Removes the forward edge from origin to target at the specified level.
    fn remove_forward_edge(&mut self, origin: usize, target: usize, level: usize) {
        let Some(origin_node) = self.graph.node_mut(origin) else {
            return;
        };

        let list = origin_node.neighbours_mut(level);
        if let Some(pos) = list.iter().position(|&id| id == target) {
            list.remove(pos);
        }
    }
}

#[inline]
fn prioritise_new_node(new_node: usize, candidates: &mut [usize]) {
    if let Some(pos) = candidates
        .iter()
        .position(|&candidate| candidate == new_node)
    {
        if pos != 0 {
            candidates.swap(0, pos);
        }
    }
}
