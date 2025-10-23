//! HNSW insertion workflow.
//!
//! Provides the planning and application phases for inserting nodes into the HNSW graph.
//! Planning computes the descent path and layer-by-layer neighbour candidates without
//! holding write locks. Application mutates the graph structure, performs bidirectional
//! linking, and schedules trimming jobs for nodes that exceed maximum connection limits.

use std::collections::{HashMap, hash_map::Entry};

use crate::DataSource;

use super::{
    error::HnswError,
    params::HnswParams,
    types::{InsertionPlan, LayerPlan},
};

use super::graph::{
    ApplyContext, DescentContext, EdgeContext, ExtendedSearchContext, Graph, LayerPlanContext,
    NodeContext, SearchContext,
};

/// Captures the neighbour candidates for a node that may require trimming.
#[derive(Clone, Debug)]
pub(super) struct TrimJob {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
}

/// Tracks the staged neighbour list for an affected node before trimming.
#[derive(Clone, Debug)]
pub(super) struct NodeUpdate {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
    pub(crate) new_node: usize,
}

impl NodeUpdate {
    fn needs_trim(&self) -> bool {
        self.candidates.len() > self.ctx.max_connections
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedInsertion {
    pub(crate) node: NodeContext,
    pub(crate) promote_entry: bool,
    pub(crate) new_node_neighbours: Vec<Vec<usize>>,
    pub(crate) updates: Vec<NodeUpdate>,
}

/// Stores the final trimmed neighbour list for a node and level.
#[derive(Clone, Debug)]
pub(super) struct TrimResult {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) neighbours: Vec<usize>,
}

impl Graph {
    /// Plans insertion of a node into the HNSW graph without mutating state.
    ///
    /// Computes the descent path from the current entry point down to the
    /// target level, then searches each layer from the target level to layer 0
    /// to identify candidate neighbours for bidirectional linking.
    ///
    /// # Parameters
    ///
    /// - `ctx` – node identifier and assigned level for the new node.
    /// - `params` – HNSW construction parameters.
    /// - `source` – data source providing distance calculations.
    ///
    /// # Errors
    ///
    /// Returns [`HnswError::GraphEmpty`] if the graph has no entry point and
    /// propagates distance or invariant failures from the descent and search
    /// phases.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let plan = graph.plan_insertion(node_ctx, &params, data_source)?;
    /// assert!(!plan.layers.is_empty());
    /// ```
    pub(crate) fn plan_insertion<D: DataSource + Sync>(
        &self,
        ctx: NodeContext,
        params: &HnswParams,
        source: &D,
    ) -> Result<InsertionPlan, HnswError> {
        let entry = self.entry().ok_or(HnswError::GraphEmpty)?;
        let target_level = ctx.level.min(entry.level);
        let descent_ctx = DescentContext {
            query: ctx.node,
            entry,
            target_level,
        };
        let current = self.greedy_descend_to_target_level(source, descent_ctx)?;
        let layer_ctx = LayerPlanContext {
            query: ctx.node,
            current,
            target_level,
            ef: params.ef_construction(),
        };
        let layers = self.build_layer_plans_from_target(source, layer_ctx)?;
        Ok(InsertionPlan { layers })
    }

    fn greedy_descend_to_target_level<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: DescentContext,
    ) -> Result<usize, HnswError> {
        let mut current = ctx.entry.node;
        if ctx.entry.level > ctx.target_level {
            for level in ((ctx.target_level + 1)..=ctx.entry.level).rev() {
                current = self.greedy_search_layer(
                    source,
                    SearchContext {
                        query: ctx.query,
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
        let mut layers = Vec::with_capacity(ctx.target_level + 1);
        let mut current = ctx.current;
        for level in (0..=ctx.target_level).rev() {
            let candidates = self.search_layer(
                source,
                ExtendedSearchContext {
                    query: ctx.query,
                    entry: current,
                    level,
                    ef: ctx.ef,
                },
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

    /// Prepares an insertion commit by staging all neighbour list updates.
    ///
    /// The returned [`PreparedInsertion`] captures the new node metadata and
    /// the neighbour lists for all affected nodes as they would appear after
    /// linking. Trimming is deferred to the caller, which should compute
    /// distances without holding the graph lock and then call
    /// [`Graph::commit_insertion`] with the resulting [`TrimResult`]s.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let (prepared, trim_jobs) = graph.apply_insertion(node_ctx, apply_ctx)?;
    /// let trims = trim_jobs
    ///     .into_iter()
    ///     .map(|job| compute_trim(job, data_source))
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// graph.commit_insertion(prepared, trims)?;
    /// ```
    pub(crate) fn apply_insertion(
        &mut self,
        node: NodeContext,
        apply_ctx: ApplyContext<'_>,
    ) -> Result<(PreparedInsertion, Vec<TrimJob>), HnswError> {
        let ApplyContext { params, plan } = apply_ctx;
        let NodeContext { node, level } = node;
        if !self.has_slot(node) {
            return Err(HnswError::InvalidParameters {
                reason: format!("node {node} is outside pre-allocated capacity"),
            });
        }
        if self.node(node).is_some() {
            return Err(HnswError::DuplicateNode { node });
        }

        let promote_entry = level > self.entry().map(|entry| entry.level).unwrap_or(0);

        let mut new_node_neighbours = vec![Vec::new(); level + 1];
        let mut updates: HashMap<(usize, usize), NodeUpdate> = HashMap::new();

        for layer in plan.layers.into_iter().filter(|layer| layer.level <= level) {
            let mut to_link = layer.neighbours;
            to_link.truncate(params.max_connections());
            let edge_ctx = EdgeContext {
                level: layer.level,
                max_connections: params.max_connections(),
            };
            self.process_layer_neighbours(
                node,
                edge_ctx,
                to_link,
                &mut new_node_neighbours,
                &mut updates,
            )?;
        }

        let updates: Vec<NodeUpdate> = updates.into_values().collect();
        let trim_jobs = self.collect_trim_jobs(&updates);

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

    fn process_layer_neighbours(
        &self,
        node: usize,
        edge_ctx: EdgeContext,
        neighbours: Vec<super::types::Neighbour>,
        new_node_neighbours: &mut [Vec<usize>],
        updates: &mut HashMap<(usize, usize), NodeUpdate>,
    ) -> Result<(), HnswError> {
        for neighbour in neighbours {
            self.process_single_neighbour(
                node,
                neighbour.id,
                edge_ctx,
                new_node_neighbours,
                updates,
            )?;
        }
        Ok(())
    }

    fn process_single_neighbour(
        &self,
        node: usize,
        neighbour_id: usize,
        edge_ctx: EdgeContext,
        new_node_neighbours: &mut [Vec<usize>],
        updates: &mut HashMap<(usize, usize), NodeUpdate>,
    ) -> Result<(), HnswError> {
        let level_neighbours = &mut new_node_neighbours[edge_ctx.level];
        if !level_neighbours.contains(&neighbour_id) {
            level_neighbours.push(neighbour_id);
        }
        let entry = match updates.entry((neighbour_id, edge_ctx.level)) {
            Entry::Occupied(existing) => existing.into_mut(),
            Entry::Vacant(vacant) => {
                let candidates = self
                    .node(neighbour_id)
                    .ok_or_else(|| HnswError::GraphInvariantViolation {
                        message: format!("node {} missing during insertion planning", neighbour_id),
                    })?
                    .neighbours(edge_ctx.level)
                    .to_vec();
                vacant.insert(NodeUpdate {
                    node: neighbour_id,
                    ctx: edge_ctx,
                    candidates,
                    new_node: node,
                })
            }
        };
        debug_assert_eq!(
            entry.new_node, node,
            "node updates must originate from the current insertion",
        );
        if !entry.candidates.contains(&node) {
            entry.candidates.push(node);
        }
        Ok(())
    }

    fn collect_trim_jobs(&self, updates: &[NodeUpdate]) -> Vec<TrimJob> {
        updates
            .iter()
            .filter(|update| update.needs_trim())
            .map(|update| {
                debug_assert!(
                    update.candidates.contains(&update.new_node),
                    "trim job missing new node candidate",
                );
                TrimJob {
                    node: update.node,
                    ctx: update.ctx,
                    candidates: reorder_candidates(update),
                }
            })
            .collect()
    }

    /// Applies a prepared insertion after trim distances have been evaluated.
    ///
    /// Consumes the [`PreparedInsertion`] generated by [`Graph::apply_insertion`]
    /// and the trimmed neighbour selections, mutating the graph in a single
    /// write-locked window. Nodes without corresponding [`TrimResult`] entries
    /// retain the staged candidate order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let (prepared, jobs) = graph.apply_insertion(node_ctx, apply_ctx)?;
    /// let trims = jobs
    ///     .into_iter()
    ///     .map(|job| compute_trim(job, data_source))
    ///     .collect::<Result<Vec<_>, _>>()?;
    /// graph.commit_insertion(prepared, trims)?;
    /// ```
    pub(crate) fn commit_insertion(
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

        self.attach_node(node.node, node.level)?;
        {
            let node_ref = self.node_mut(node.node).expect("node was attached above");
            for (level, neighbours) in new_node_neighbours.into_iter().enumerate() {
                if level > node.level {
                    break;
                }
                let list = node_ref.neighbours_mut(level);
                list.clear();
                list.extend(neighbours);
            }
        }
        if promote_entry {
            self.promote_entry(node.node, node.level);
        }

        let mut trim_lookup: HashMap<(usize, usize), Vec<usize>> = trims
            .into_iter()
            .map(|result| ((result.node, result.ctx.level), result.neighbours))
            .collect();

        for update in updates {
            let neighbours = trim_lookup
                .remove(&(update.node, update.ctx.level))
                .unwrap_or_else(|| update.candidates.clone());
            let node_ref =
                self.node_mut(update.node)
                    .ok_or_else(|| HnswError::GraphInvariantViolation {
                        message: format!("node {} missing during insertion commit", update.node),
                    })?;
            let list = node_ref.neighbours_mut(update.ctx.level);
            list.clear();
            list.extend(neighbours);
        }
        Ok(())
    }
}

/// Reorders candidates so the new node is validated before existing edges.
fn reorder_candidates(update: &NodeUpdate) -> Vec<usize> {
    let mut candidates = update.candidates.clone();
    if let Some(pos) = candidates
        .iter()
        .position(|&candidate| candidate == update.new_node)
    {
        if pos != 0 {
            candidates.swap(0, pos);
        }
    }
    candidates
}
