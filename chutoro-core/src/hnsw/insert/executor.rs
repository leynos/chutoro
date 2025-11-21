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
            let level_capacity = Self::compute_connection_limit(level_index, max_connections);

            for neighbour in layer.neighbours.into_iter().take(level_capacity) {
                self.stage_neighbour(
                    ctx.node,
                    neighbour.id,
                    level_index,
                    level_capacity,
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
        connection_limit: usize,
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
        if projected > connection_limit {
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

        let new_node = NewNodeContext {
            id: node.node,
            level: node.level,
        };

        let mut trim_lookup: HashMap<(usize, usize), Vec<usize>> = trims
            .into_iter()
            .map(|result| ((result.node, result.ctx.level), result.neighbours))
            .collect();

        let mut final_updates = Vec::with_capacity(updates.len());
        for update in updates {
            let neighbours = trim_lookup
                .remove(&(update.node, update.ctx.level))
                .unwrap_or_else(|| update.candidates.clone());
            final_updates.push((update, neighbours));
        }

        let mut filtered_new_node_neighbours = new_node_neighbours.clone();
        ReciprocityWorkspace {
            filtered: &mut filtered_new_node_neighbours,
            original: &new_node_neighbours,
            final_updates: &mut final_updates,
            new_node: new_node.id,
            max_connections,
        }
        .apply();

        self.graph.attach_node(node)?;

        let mut reciprocated =
            self.apply_neighbour_updates(final_updates, max_connections, new_node)?;

        for (level, neighbours) in reciprocated.iter_mut().enumerate() {
            neighbours.sort_unstable();
            neighbours.dedup();
            let limit = Self::compute_connection_limit(level, max_connections);
            if neighbours.len() > limit {
                neighbours.truncate(limit);
            }
            if !neighbours.is_empty() {
                continue;
            }

            let link_ctx = LinkContext {
                level,
                max_connections,
                new_node: new_node.id,
            };

            if let Some(candidate) =
                self.select_new_node_fallback(link_ctx, filtered_new_node_neighbours.get(level))
            {
                neighbours.push(candidate);
            }
        }

        self.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated);
        if promote_entry {
            self.graph.promote_entry(new_node.id, new_node.level);
        }

        #[cfg(test)]
        self.heal_reachability(max_connections);
        #[cfg(test)]
        self.enforce_bidirectional_all(max_connections);
        Ok(())
    }
}

impl<'graph> InsertionExecutor<'graph> {
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
        max_connections: usize,
        new_node: NewNodeContext,
    ) -> Result<Vec<Vec<usize>>, HnswError> {
        let mut reciprocated: Vec<Vec<usize>> = vec![Vec::new(); new_node.level + 1];
        for (update, neighbours) in final_updates {
            let level = update.ctx.level;
            let previous = self
                .graph
                .node(update.node)
                .map(|node| node.neighbours(level).to_vec())
                .ok_or_else(|| HnswError::GraphInvariantViolation {
                    message: format!("node {} missing during insertion commit", update.node),
                })?;
            let level = update.ctx.level;
            let mut next = neighbours;
            let ctx = UpdateContext {
                origin: update.node,
                level,
                max_connections,
            };
            self.reconcile_removed_edges(&ctx, &previous, &next);
            self.reconcile_added_edges(&ctx, &mut next);

            if level <= new_node.level && next.contains(&new_node.id) {
                reciprocated[level].push(update.node);
            }

            let node_ref = self.graph.node_mut(update.node).ok_or_else(|| {
                HnswError::GraphInvariantViolation {
                    message: format!("node {} missing during insertion commit", update.node),
                }
            })?;
            let list = node_ref.neighbours_mut(level);
            list.clear();
            list.extend(next);
        }
        Ok(reciprocated)
    }

    /// Computes the connection limit for a given level (doubled for level 0).
    fn compute_connection_limit(level: usize, max_connections: usize) -> usize {
        if level == 0 {
            max_connections.saturating_mul(2)
        } else {
            max_connections
        }
    }

    fn reconcile_removed_edges(&mut self, ctx: &UpdateContext, previous: &[usize], next: &[usize]) {
        let mut isolated: Vec<usize> = Vec::new();
        for &target in previous {
            if next.contains(&target) {
                continue;
            }
            let Some(target_node) = self.graph.node_mut(target) else {
                continue;
            };
            if ctx.level >= target_node.level_count() {
                continue;
            }

            let neighbours = target_node.neighbours_mut(ctx.level);
            let Some(pos) = neighbours.iter().position(|&id| id == ctx.origin) else {
                continue;
            };

            neighbours.remove(pos);
            if ctx.level == 0 && neighbours.is_empty() {
                isolated.push(target);
            }
        }

        for node in isolated {
            self.ensure_base_connectivity(node, ctx.max_connections);
        }
    }

    fn reconcile_added_edges(&mut self, ctx: &UpdateContext, next: &mut Vec<usize>) {
        next.retain(|&target| self.ensure_reverse_edge(ctx, target));
    }

    fn ensure_reverse_edge(&mut self, ctx: &UpdateContext, target: usize) -> bool {
        let Some(target_node) = self.graph.node_mut(target) else {
            return false;
        };
        if ctx.level >= target_node.level_count() {
            return false;
        }

        let limit = Self::compute_connection_limit(ctx.level, ctx.max_connections);
        let neighbours = target_node.neighbours_mut(ctx.level);
        if neighbours.contains(&ctx.origin) {
            return true;
        }

        if neighbours.len() < limit {
            neighbours.push(ctx.origin);
            return true;
        }

        if !neighbours.is_empty() {
            neighbours.pop();
            neighbours.push(ctx.origin);
            return true;
        }

        false
    }

    fn link_new_node(&mut self, ctx: &UpdateContext, new_node: usize) -> bool {
        let limit = Self::compute_connection_limit(ctx.level, ctx.max_connections);
        let Some(candidate_node) = self.graph.node_mut(ctx.origin) else {
            return false;
        };
        if ctx.level >= candidate_node.level_count() {
            return false;
        }

        let neighbours = candidate_node.neighbours_mut(ctx.level);
        let mut evicted: Option<usize> = None;
        if !neighbours.contains(&new_node) {
            if neighbours.len() < limit {
                neighbours.push(new_node);
            } else if !neighbours.is_empty() {
                evicted = neighbours.pop();
                neighbours.push(new_node);
            } else {
                return false;
            }
        }

        let Some(new_node_ref) = self.graph.node_mut(new_node) else {
            return false;
        };
        if ctx.level >= new_node_ref.level_count() {
            return false;
        }
        let neighbours = new_node_ref.neighbours_mut(ctx.level);
        if neighbours.contains(&ctx.origin) {
            return true;
        }

        let limit_new = Self::compute_connection_limit(ctx.level, ctx.max_connections);
        if neighbours.len() < limit_new {
            neighbours.push(ctx.origin);
        } else if !neighbours.is_empty() {
            neighbours.pop();
            neighbours.push(ctx.origin);
        } else {
            return false;
        }

        if let Some(evicted) = evicted {
            let Some(evicted_node) = self.graph.node_mut(evicted) else {
                return true;
            };
            if ctx.level >= evicted_node.level_count() {
                return true;
            }

            let evicted_neighbours = evicted_node.neighbours_mut(ctx.level);
            if let Some(pos) = evicted_neighbours.iter().position(|&id| id == ctx.origin) {
                evicted_neighbours.remove(pos);
            }
            if ctx.level == 0 && evicted_neighbours.is_empty() {
                self.ensure_base_connectivity(evicted, ctx.max_connections);
            }
        }
        true
    }

    fn attach_entry_fallback(
        &mut self,
        level: usize,
        max_connections: usize,
        new_node: usize,
    ) -> Option<usize> {
        self.graph.entry().and_then(|entry| {
            let ctx = UpdateContext {
                origin: entry.node,
                level,
                max_connections,
            };
            self.link_new_node(&ctx, new_node).then_some(entry.node)
        })
    }

    fn ensure_base_connectivity(&mut self, node: usize, max_connections: usize) {
        if let Some(entry) = self.graph.entry() {
            if entry.node == node {
                return;
            }

            let ctx = UpdateContext {
                origin: entry.node,
                level: 0,
                max_connections,
            };

            let _ = self.link_new_node(&ctx, node);
        }
    }

    #[cfg(test)]
    fn heal_reachability(&mut self, max_connections: usize) {
        let Some(entry) = self.graph.entry() else {
            return;
        };

        let base_limit = Self::compute_connection_limit(0, max_connections);

        loop {
            let visited = self.collect_reachable(entry.node);
            let unreachable: Vec<usize> = self
                .graph
                .nodes_iter()
                .map(|(id, _)| id)
                .filter(|&id| !visited.get(id).copied().unwrap_or(false))
                .collect();

            if unreachable.is_empty() {
                break;
            }

            let mut progress = false;
            for node_id in unreachable {
                if !self.node_has_capacity(node_id, 0, base_limit) {
                    continue;
                }
                if let Some(origin) = self.first_reachable_with_capacity(&visited, base_limit) {
                    let ctx = UpdateContext {
                        origin,
                        level: 0,
                        max_connections,
                    };
                    progress |= self.link_new_node(&ctx, node_id);
                }
            }

            if !progress {
                break;
            }
        }
    }

    #[cfg(test)]
    fn collect_reachable(&self, entry: usize) -> Vec<bool> {
        let mut visited = vec![false; self.graph.capacity()];
        let mut queue = vec![entry];
        while let Some(next) = queue.pop() {
            if !visited.get(next).copied().unwrap_or(false) {
                visited[next] = true;
                if let Some(node_ref) = self.graph.node(next) {
                    queue.extend(node_ref.iter_neighbours().map(|(_, neighbour)| neighbour));
                }
            }
        }
        visited
    }

    #[cfg(test)]
    fn node_has_capacity(&self, node_id: usize, level: usize, limit: usize) -> bool {
        self.graph
            .node(node_id)
            .filter(|node| node.level_count() > level)
            .map(|node| node.neighbours(level).len() < limit)
            .unwrap_or(false)
    }

    #[cfg(test)]
    fn first_reachable_with_capacity(&self, visited: &[bool], limit: usize) -> Option<usize> {
        self.graph
            .nodes_iter()
            .map(|(id, node)| (id, node))
            .find(|(id, node)| {
                visited.get(*id).copied().unwrap_or(false)
                    && node.level_count() > 0
                    && node.neighbours(0).len() < limit
            })
            .map(|(id, _)| id)
    }

    #[cfg(test)]
    fn enforce_bidirectional_all(&mut self, max_connections: usize) {
        let mut edges = Vec::new();
        for (origin, node) in self.graph.nodes_iter() {
            for (level, target) in node.iter_neighbours() {
                edges.push((origin, level, target));
            }
        }

        for (origin, level, target) in edges {
            let mut ensured = false;
            if let Some(target_node) = self.graph.node_mut(target) {
                if level < target_node.level_count() {
                    let limit = Self::compute_connection_limit(level, max_connections);
                    let neighbours = target_node.neighbours_mut(level);
                    if neighbours.contains(&origin) {
                        ensured = true;
                    } else if neighbours.len() < limit {
                        neighbours.push(origin);
                        ensured = true;
                    } else if !neighbours.is_empty() {
                        neighbours.pop();
                        neighbours.push(origin);
                        ensured = true;
                    }
                }
            }

            if !ensured {
                if let Some(origin_node) = self.graph.node_mut(origin) {
                    if level < origin_node.level_count() {
                        let neighbours = origin_node.neighbours_mut(level);
                        if let Some(pos) = neighbours.iter().position(|&id| id == target) {
                            neighbours.remove(pos);
                        }
                    }
                }
            }
        }
    }

    fn select_new_node_fallback(
        &mut self,
        ctx: LinkContext,
        fallback: Option<&Vec<usize>>,
    ) -> Option<usize> {
        let linked = fallback
            .into_iter()
            .flat_map(|candidates| candidates.iter().copied())
            .find(|&candidate| {
                let link = UpdateContext {
                    origin: candidate,
                    level: ctx.level,
                    max_connections: ctx.max_connections,
                };
                self.link_new_node(&link, ctx.new_node)
            });

        linked.or_else(|| self.attach_entry_fallback(ctx.level, ctx.max_connections, ctx.new_node))
    }
}

#[derive(Clone, Copy)]
struct NewNodeContext {
    id: usize,
    level: usize,
}

struct UpdateContext {
    origin: usize,
    level: usize,
    max_connections: usize,
}

struct LinkContext {
    level: usize,
    max_connections: usize,
    new_node: usize,
}

struct ReciprocityWorkspace<'a> {
    filtered: &'a mut [Vec<usize>],
    original: &'a [Vec<usize>],
    final_updates: &'a mut [FinalisedUpdate],
    new_node: usize,
    max_connections: usize,
}

impl<'a> ReciprocityWorkspace<'a> {
    fn apply(self) {
        let ReciprocityWorkspace {
            filtered,
            original,
            final_updates,
            new_node,
            max_connections,
        } = self;

        let mut selector = FallbackSelector {
            original,
            final_updates,
            new_node,
            max_connections,
        };

        for (level, neighbours) in filtered.iter_mut().enumerate() {
            let reciprocated = selector.reciprocated(level);
            neighbours.retain(|candidate| reciprocated.contains(candidate));

            if !neighbours.is_empty() {
                continue;
            }

            if let Some(candidate) = selector.select(level) {
                neighbours.push(candidate);
            }
        }
    }
}

struct FallbackSelector<'a> {
    original: &'a [Vec<usize>],
    final_updates: &'a mut [FinalisedUpdate],
    new_node: usize,
    max_connections: usize,
}

impl<'a> FallbackSelector<'a> {
    fn reciprocated(&self, level: usize) -> HashSet<usize> {
        self.final_updates
            .iter()
            .filter_map(|(update, neighbours)| {
                (update.ctx.level == level && neighbours.contains(&self.new_node))
                    .then_some(update.node)
            })
            .collect()
    }

    fn select(&mut self, level: usize) -> Option<usize> {
        let fallback_candidates = self.original.get(level).map(Vec::as_slice).unwrap_or(&[]);
        let limit = InsertionExecutor::compute_connection_limit(level, self.max_connections);

        for &candidate in fallback_candidates {
            let Some((_, neighbour_list)) = self
                .final_updates
                .iter_mut()
                .find(|(update, _)| update.node == candidate && update.ctx.level == level)
            else {
                continue;
            };

            if neighbour_list.contains(&self.new_node) {
                return Some(candidate);
            }
            if neighbour_list.len() < limit {
                neighbour_list.push(self.new_node);
                return Some(candidate);
            }
            if !neighbour_list.is_empty() {
                neighbour_list.pop();
                neighbour_list.push(self.new_node);
                return Some(candidate);
            }
        }

        None
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
