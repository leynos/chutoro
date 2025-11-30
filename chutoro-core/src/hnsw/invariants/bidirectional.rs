//! Bidirectional edge invariant for the CPU HNSW graph.
//!
//! Ensures every directed link has a matching reverse link on the same layer,
//! preserving the mutual neighbourhood guarantee required for search and
//! insertion correctness.
use super::{
    EvaluationMode, GraphContext, HnswInvariantViolation,
    helpers::{LayerValidator, for_each_edge},
};
use tracing::{Level, debug, trace};

pub(super) fn check_bidirectional(
    ctx: GraphContext<'_>,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    let validator = LayerValidator::new(ctx.graph);
    if tracing::enabled!(Level::TRACE) {
        let edge_count: usize = ctx
            .graph
            .nodes_iter()
            .map(|(_, node)| node.iter_neighbours().count())
            .sum();
        trace!(edges = edge_count, "checking bidirectional links");
    }
    for_each_edge(ctx.graph, |source, target, level| {
        trace!(source, target, level, "checking edge for backlink");
        match validator.ensure(source, target, level) {
            Ok(neighbour) => {
                let neighbours = neighbour.neighbours(level);
                if neighbours.contains(&source) {
                    Ok(())
                } else {
                    debug!(
                        source,
                        target,
                        level,
                        neighbours = ?neighbours,
                        "missing backlink"
                    );
                    mode.record(HnswInvariantViolation::MissingBacklink {
                        origin: source,
                        target,
                        layer: level,
                    })
                }
            }
            Err(err) => mode.record(err),
        }
    })
}
