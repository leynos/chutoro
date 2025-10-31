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
/// use chutoro_core::hnsw::insert::executor::{EdgeContext, TrimJob};
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

        let contains_new = candidates.iter().any(|&existing| existing == new_node);
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
        } = prepared;

        self.graph.attach_node(node)?;
        {
            let node_ref = self
                .graph
                .node_mut(node.node)
                .expect("node was attached above");
            for (level, neighbours) in new_node_neighbours
                .into_iter()
                .enumerate()
                .take(node.level + 1)
            {
                let list = node_ref.neighbours_mut(level);
                list.clear();
                list.extend(neighbours);
            }
        }
        if promote_entry {
            self.graph.promote_entry(node.node, node.level);
        }

        let mut trim_lookup: HashMap<(usize, usize), Vec<usize>> = trims
            .into_iter()
            .map(|result| ((result.node, result.ctx.level), result.neighbours))
            .collect();

        for update in updates {
            let neighbours = trim_lookup
                .remove(&(update.node, update.ctx.level))
                .unwrap_or_else(|| update.candidates.clone());
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
