//! Plans HNSW insertions without mutating the graph.

use crate::{
    DataSource,
    hnsw::{
        error::HnswError,
        graph::{DescentContext, Graph, LayerPlanContext, NodeContext, SearchContext},
        params::HnswParams,
        types::{InsertionPlan, LayerPlan},
    },
};

#[derive(Debug)]
pub(crate) struct InsertionPlanner<'graph> {
    graph: &'graph Graph,
}

impl<'graph> InsertionPlanner<'graph> {
    pub(crate) fn new(graph: &'graph Graph) -> Self {
        Self { graph }
    }

    /// Plans insertion of a node into the HNSW graph without mutating state.
    ///
    /// Computes the descent path from the current entry point down to the
    /// target level, then searches each layer from the target level to layer 0
    /// to identify candidate neighbours for bidirectional linking.
    pub(crate) fn plan<D: DataSource + Sync>(
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
        if ctx.entry.level > ctx.target_level {
            let searcher = self.graph.searcher();
            for level in ((ctx.target_level + 1)..=ctx.entry.level).rev() {
                current = searcher.greedy_search_layer(
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
        let searcher = self.graph.searcher();
        for level in (0..=ctx.target_level).rev() {
            let candidates = searcher.search_layer(
                source,
                SearchContext {
                    query: ctx.query,
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
