use std::collections::HashSet;

use crate::DataSource;

use super::{
    error::HnswError,
    params::HnswParams,
    types::{InsertionPlan, LayerPlan},
};

use super::graph::{
    ApplyContext, DescentContext, EdgeContext, ExtendedSearchContext, Graph, LayerPlanContext,
    NodeContext, NodePair, SearchContext,
};

#[derive(Clone, Debug)]
pub(super) struct TrimJob {
    pub(crate) node: usize,
    pub(crate) ctx: EdgeContext,
    pub(crate) candidates: Vec<usize>,
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

    pub(crate) fn apply_insertion(
        &mut self,
        node: NodeContext,
        apply_ctx: ApplyContext<'_>,
    ) -> Result<Vec<TrimJob>, HnswError> {
        let ApplyContext { params, plan } = apply_ctx;
        let NodeContext { node, level } = node;
        self.attach_node(node, level)?;
        self.promote_entry(node, level);

        let mut trim_jobs = Vec::new();

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
                if let Some(job) = self.prepare_trim_job(changed, edge_ctx)? {
                    trim_jobs.push(job);
                }
            }
        }
        Ok(trim_jobs)
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

    fn prepare_trim_job(
        &self,
        node: usize,
        ctx: EdgeContext,
    ) -> Result<Option<TrimJob>, HnswError> {
        let node_ref = self
            .node(node)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!("node {node} missing during trim scheduling"),
            })?;
        let neighbours = node_ref.neighbours(ctx.level);
        if neighbours.len() <= ctx.max_connections {
            return Ok(None);
        }
        Ok(Some(TrimJob {
            node,
            ctx,
            candidates: neighbours.to_vec(),
        }))
    }

    pub(crate) fn apply_trim(
        &mut self,
        node: usize,
        ctx: EdgeContext,
        scored: &[(usize, f32)],
    ) -> Result<(), HnswError> {
        debug_assert!(
            scored.len() <= ctx.max_connections,
            "trimmed candidate list exceeds max connections"
        );
        let node_ref = self
            .node_mut(node)
            .ok_or_else(|| HnswError::GraphInvariantViolation {
                message: format!("node {node} missing during trim"),
            })?;
        let list = node_ref.neighbours_mut(ctx.level);
        list.clear();
        list.extend(scored.iter().map(|(candidate, _)| *candidate));
        Ok(())
    }
}
