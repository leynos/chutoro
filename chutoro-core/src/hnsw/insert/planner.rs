//! Plans HNSW insertions without mutating the graph.

use crate::{
    DataSource,
    hnsw::{
        distance_cache::DistanceCache,
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

/// Inputs required to plan an insertion without mutating the graph.
///
/// Provide the node context, HNSW parameters, the data source, and an optional
/// distance cache used for greedy descent and layer searches.
///
/// # Examples
/// ```rust,ignore
/// use crate::hnsw::{DistanceCacheConfig, HnswParams};
/// use crate::hnsw::graph::NodeContext;
/// use crate::hnsw::insert::planner::PlanningInputs;
/// use chutoro_core::DataSource;
/// use std::num::NonZeroUsize;
///
/// struct DummySource;
///
/// impl DataSource for DummySource {
///     fn len(&self) -> usize { 1 }
///     fn name(&self) -> &str { "dummy" }
///     fn distance(&self, _: usize, _: usize) -> Result<f32, chutoro_core::DataSourceError> {
///         Ok(0.0)
///     }
///     fn batch_distances(&self, _: usize, _: &[usize]) -> Result<Vec<f32>, chutoro_core::DataSourceError> {
///         Ok(vec![0.0])
///     }
///     fn metric_descriptor(&self) -> chutoro_core::MetricDescriptor {
///         chutoro_core::MetricDescriptor::new("dummy")
///     }
/// }
///
/// let params = HnswParams::new(8, 16).unwrap()
///     .with_distance_cache_config(DistanceCacheConfig::new(NonZeroUsize::new(128).unwrap()));
/// let ctx = NodeContext { node: 0, level: 0, sequence: 0 };
/// let source = DummySource;
/// let inputs = PlanningInputs {
///     ctx,
///     params: &params,
///     source: &source,
///     cache: None,
/// };
/// assert_eq!(inputs.ctx.node, 0);
/// ```
#[derive(Clone, Copy)]
pub(crate) struct PlanningInputs<'a, D: DataSource + Sync> {
    pub(crate) ctx: NodeContext,
    pub(crate) params: &'a HnswParams,
    pub(crate) source: &'a D,
    pub(crate) cache: Option<&'a DistanceCache>,
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
        inputs: PlanningInputs<'_, D>,
    ) -> Result<InsertionPlan, HnswError> {
        let PlanningInputs {
            ctx,
            params,
            source,
            cache,
        } = inputs;
        let entry = self.graph.entry().ok_or(HnswError::GraphEmpty)?;
        let target_level = ctx.level.min(entry.level);
        let descent_ctx = DescentContext::new(ctx.node, entry, target_level);
        let current = self.greedy_descend_to_target_level(source, descent_ctx, cache)?;
        let layer_ctx =
            LayerPlanContext::new(ctx.node, current, target_level, params.ef_construction());
        let layers = self.build_layer_plans_from_target(source, layer_ctx, cache)?;
        Ok(InsertionPlan { layers })
    }

    fn greedy_descend_to_target_level<D: DataSource + Sync>(
        &self,
        source: &D,
        ctx: DescentContext,
        cache: Option<&DistanceCache>,
    ) -> Result<usize, HnswError> {
        let mut current = ctx.entry.node;
        if ctx.entry.level > ctx.target_level {
            let searcher = self.graph.searcher();
            for level in ((ctx.target_level + 1)..=ctx.entry.level).rev() {
                current = searcher.greedy_search_layer(
                    cache,
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
        cache: Option<&DistanceCache>,
    ) -> Result<Vec<LayerPlan>, HnswError> {
        let mut layers = Vec::with_capacity(ctx.target_level + 1);
        let mut current = ctx.current;
        let searcher = self.graph.searcher();
        for level in (0..=ctx.target_level).rev() {
            let candidates = searcher.search_layer(
                cache,
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
