//! HNSW insertion workflow.
//!
//! Provides the planning and application phases for inserting nodes into the HNSW graph.
//! Planning computes the descent path and layer-by-layer neighbour candidates without
//! holding write locks. Application mutates the graph structure, performs bidirectional
//! linking, and schedules trimming jobs for nodes that exceed maximum connection limits.

use std::collections::{HashMap, HashSet, hash_map::Entry};

use crate::DataSource;

use super::{
    error::HnswError,
    params::HnswParams,
    types::{InsertionPlan, LayerPlan},
};

use super::graph::{
    ApplyContext, DescentContext, EdgeContext, Graph, LayerPlanContext, NodeContext, SearchContext,
};

/// Captures the neighbour candidates for a node that may require trimming.
#[derive(Clone, Debug)]
pub(super) struct TrimJob {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedInsertion {
    pub(crate) node: NodeContext,
    pub(crate) promote_entry: bool,
    pub(crate) new_node_neighbours: Vec<Vec<usize>>,
    pub(crate) updates: Vec<StagedUpdate>,
}

/// Captures the staged neighbour set for a node at a given level.
#[derive(Clone, Debug)]
pub(super) struct StagedUpdate {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
}

/// Stores the final trimmed neighbour list for a node and level.
#[derive(Clone, Debug)]
pub(super) struct TrimResult {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) neighbours: Vec<usize>,
}

#[derive(Debug)]
pub(super) struct InsertionPlanner<'graph> {
    graph: &'graph Graph,
}

impl<'graph> InsertionPlanner<'graph> {
    pub(super) fn new(graph: &'graph Graph) -> Self {
        Self { graph }
    }

    /// Plans insertion of a node into the HNSW graph without mutating state.
    ///
    /// Computes the descent path from the current entry point down to the
    /// target level, then searches each layer from the target level to layer 0
    /// to identify candidate neighbours for bidirectional linking.
    pub(super) fn plan<D: DataSource + Sync>(
        &self,
        ctx: NodeContext,
        params: &HnswParams,
        source: &D,
    ) -> Result<InsertionPlan, HnswError> {
        let entry = self.graph.entry().ok_or(HnswError::GraphEmpty)?;
        let target_level = ctx.level.min(entry.level);
        let descent_ctx = DescentContext::new(ctx.node, entry, target_level);
        let current = self.greedy_descend_to_target_level(source, descent_ctx)?;
        let layer_ctx =
            LayerPlanContext::new(ctx.node, current, target_level, params.ef_construction());
        let layers = self.build_layer_plans_from_target(source, layer_ctx)?;
        Ok(InsertionPlan { layers })
    }

    fn greedy_descend_to_target_level<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: DescentContext,
    ) -> Result<usize, HnswError> {
        let mut current = ctx.entry.node;
        if ctx.entry.level > ctx.target_level() {
            let searcher = self.graph.searcher();
            for level in ((ctx.target_level() + 1)..=ctx.entry.level).rev() {
                current = searcher.greedy_search_layer(
                    source,
                    SearchContext {
                        query: ctx.query(),
                        entry: current,
                        level,
                    },
                )?;
            }
        }
        Ok(current)
    }

    fn build_layer_plans_from_target<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: LayerPlanContext,
    ) -> Result<Vec<LayerPlan>, HnswError> {
        let mut layers = Vec::with_capacity(ctx.target_level() + 1);
        let mut current = ctx.current;
        let searcher = self.graph.searcher();
        for level in (0..=ctx.target_level()).rev() {
            let candidates = searcher.search_layer(
                source,
                SearchContext {
                    query: ctx.query(),
                    entry: current,
                    level,
                }
                .with_ef(ctx.ef),
            )?;
            if let Some(best) = candidates.first() {
                current = best.id;
            }
            layers.push(LayerPlan {
                level,
                neighbours: candidates,
            });
        }
        layers.reverse();
        Ok(layers)
    }
}

#[derive(Debug)]
pub(super) struct InsertionExecutor<'graph> {
    graph: &'graph mut Graph,
}

impl<'graph> InsertionExecutor<'graph> {
    pub(super) fn new(graph: &'graph mut Graph) -> Self {
        Self { graph }
    }

    /// Prepares an insertion commit by staging all neighbour list updates.
    ///
    /// The returned [`PreparedInsertion`] captures the new node metadata and
    /// the neighbour lists for all affected nodes as they would appear after
    /// linking. Trimming is deferred to the caller, which should compute
    /// distances without holding the graph lock and then call
    /// [`InsertionExecutor::commit`] with the resulting [`TrimResult`]s.
    pub(super) fn apply(
        &mut self,
        node: NodeContext,
        apply_ctx: ApplyContext<'_>,
    ) -> Result<(PreparedInsertion, Vec<TrimJob>), HnswError> {
        let ApplyContext { params, plan } = apply_ctx;
        let NodeContext { node, level } = node;
        if !self.graph.has_slot(node) {
            return Err(HnswError::InvalidParameters {
                reason: format!("node {node} is outside pre-allocated capacity"),
            });
        }
        if self.graph.node(node).is_some() {
            return Err(HnswError::DuplicateNode { node });
        }

        let promote_entry = level > self.graph.entry().map(|entry| entry.level).unwrap_or(0);
        let max_connections = params.max_connections();

        let mut new_node_neighbours = vec![Vec::new(); level + 1];
        let mut staged: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut needs_trim = HashSet::new();

        for layer in plan.layers.into_iter().filter(|layer| layer.level <= level) {
            let edge_ctx = EdgeContext {
                level: layer.level,
                max_connections,
            };

            for neighbour in layer.neighbours.into_iter().take(max_connections) {
                let level_neighbours = &mut new_node_neighbours[edge_ctx.level];
                Self::record_new_node_link(level_neighbours, neighbour.id);

                let candidates =
                    self.ensure_staged_candidates(&mut staged, neighbour.id, edge_ctx.level)?;
                Self::ensure_reciprocal_link(candidates, node);

                Self::mark_trim_requirement(
                    &mut needs_trim,
                    candidates.len(),
                    max_connections,
                    (neighbour.id, edge_ctx.level),
                );
            }
        }

        let mut updates = Vec::with_capacity(staged.len());
        let mut trim_jobs = Vec::with_capacity(needs_trim.len());

        for ((other, lvl), candidates) in staged.into_iter() {
            let ctx = EdgeContext {
                level: lvl,
                max_connections,
            };
            if needs_trim.contains(&(other, lvl)) {
                let mut reordered = candidates.clone();
                prioritise_new_node(node, &mut reordered);
                trim_jobs.push(TrimJob {
                    node: other,
                    ctx,
                    candidates: reordered,
                });
            }
            updates.push(StagedUpdate {
                node: other,
                ctx,
                candidates,
            });
        }

        Ok((
            PreparedInsertion {
                node: NodeContext { node, level },
                promote_entry,
                new_node_neighbours,
                updates,
            },
            trim_jobs,
        ))
    }

    fn ensure_staged_candidates<'a>(
        &self,
        staged: &'a mut HashMap<(usize, usize), Vec<usize>>,
        neighbour: usize,
        level: usize,
    ) -> Result<&'a mut Vec<usize>, HnswError> {
        let entry = staged.entry((neighbour, level));
        match entry {
            Entry::Occupied(existing) => Ok(existing.into_mut()),
            Entry::Vacant(vacant) => Ok(vacant.insert(self.initial_candidates(neighbour, level)?)),
        }
    }

    fn initial_candidates(&self, node: usize, level: usize) -> Result<Vec<usize>, HnswError> {
        let Some(graph_node) = self.graph.node(node) else {
            return Err(HnswError::GraphInvariantViolation {
                message: format!("node {node} missing during insertion planning at level {level}",),
            });
        };
        Ok(graph_node.neighbours(level).to_vec())
    }

    fn ensure_reciprocal_link(candidates: &mut Vec<usize>, node: usize) {
        if candidates.contains(&node) {
            return;
        }
        candidates.push(node);
    }

    fn record_new_node_link(level_neighbours: &mut Vec<usize>, candidate: usize) {
        if level_neighbours.contains(&candidate) {
            return;
        }
        level_neighbours.push(candidate);
    }

    fn mark_trim_requirement(
        needs_trim: &mut HashSet<(usize, usize)>,
        candidates_len: usize,
        max_connections: usize,
        key: (usize, usize),
    ) {
        if candidates_len > max_connections {
            needs_trim.insert(key);
        }
    }

    /// Applies a prepared insertion after trim distances have been evaluated.
    pub(super) fn commit(
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

        self.graph.attach_node(node.node, node.level)?;
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
