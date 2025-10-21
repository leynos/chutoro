use std::{cmp::Ordering, collections::HashSet};

use crate::DataSource;

use super::{
    error::HnswError,
    params::HnswParams,
    search::{ExtendedSearchContext, SearchContext},
    types::{EntryPoint, InsertionPlan, LayerPlan},
    validate::validate_batch_distances,
};

use super::graph::Graph;

#[derive(Clone, Copy, Debug)]
pub(crate) struct NodeContext {
    pub(crate) node: usize,
    pub(crate) level: usize,
}

#[derive(Clone, Copy, Debug)]
struct EdgeContext {
    level: usize,
    max_connections: usize,
}

#[derive(Clone, Copy, Debug)]
struct DescentContext {
    query: usize,
    entry: EntryPoint,
    target_level: usize,
}

#[derive(Clone, Copy, Debug)]
struct LayerPlanContext {
    query: usize,
    current: usize,
    target_level: usize,
    ef: usize,
}

#[derive(Clone, Copy, Debug)]
struct NodePair {
    from: usize,
    to: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ApplyContext<'a> {
    pub(crate) params: &'a HnswParams,
    pub(crate) plan: InsertionPlan,
}

impl Graph {
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

    pub(crate) fn apply_insertion<D: DataSource + Sync>(
        &mut self,
        node: NodeContext,
        apply_ctx: ApplyContext<'_>,
        source: &D,
    ) -> Result<(), HnswError> {
        let ApplyContext { params, plan } = apply_ctx;
        let NodeContext { node, level } = node;
        self.attach_node(node, level)?;
        self.promote_entry(node, level);

        for layer in plan.layers.into_iter().filter(|layer| layer.level <= level) {
            let mut to_link = layer.neighbours;
            to_link.truncate(params.max_connections());
            let edge_ctx = EdgeContext {
                level: layer.level,
                max_connections: params.max_connections(),
            };
            let mut trim_targets = HashSet::new();
            for neighbour in to_link {
                let updated = self.link_bidirectional(
                    NodePair {
                        from: node,
                        to: neighbour.id,
                    },
                    edge_ctx,
                )?;
                for changed in updated {
                    trim_targets.insert(changed);
                }
            }
            for changed in trim_targets {
                self.trim_neighbours(changed, edge_ctx, source)?;
            }
        }
        Ok(())
    }

    fn link_bidirectional(
        &mut self,
        pair: NodePair,
        ctx: EdgeContext,
    ) -> Result<Vec<usize>, HnswError> {
        let mut updated = Vec::with_capacity(2);
        if self.link_one_way(pair, ctx)? {
            updated.push(pair.from);
        }
        if self.link_one_way(
            NodePair {
                from: pair.to,
                to: pair.from,
            },
            ctx,
        )? {
            updated.push(pair.to);
        }
        Ok(updated)
    }

    fn link_one_way(&mut self, pair: NodePair, ctx: EdgeContext) -> Result<bool, HnswError> {
        let node = self
            .node_mut(pair.from)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!("node {} missing during link", pair.from),
            })?;
        let list = node.neighbours_mut(ctx.level);
        if !list.contains(&pair.to) {
            list.push(pair.to);
            return Ok(true);
        }
        Ok(false)
    }

    fn trim_neighbours<D: DataSource + Sync>(
        &mut self,
        node: usize,
        ctx: EdgeContext,
        source: &D,
    ) -> Result<(), HnswError> {
        let candidates = {
            let node_ref =
                self.node_mut(node)
                    .ok_or_else(|| HnswError::GraphInvariantViolation {
                        message: format!("node {node} missing during trim"),
                    })?;
            let list = node_ref.neighbours_mut(ctx.level);
            if list.len() <= ctx.max_connections {
                return Ok(());
            }
            list.clone()
        };

        let distances = validate_batch_distances(source, node, &candidates)?;
        let mut scored: Vec<_> = candidates.into_iter().zip(distances).collect();
        scored.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        scored.truncate(ctx.max_connections);

        let node_ref = self
            .node_mut(node)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!("node {node} missing during trim"),
            })?;
        let list = node_ref.neighbours_mut(ctx.level);
        list.clear();
        list.extend(scored.into_iter().map(|(candidate, _)| candidate));
        Ok(())
    }
}
