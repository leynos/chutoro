//! Bidirectional edge invariant for the CPU HNSW graph.
//!
//! Ensures every directed link has a matching reverse link on the same layer,
//! preserving the mutual neighbourhood guarantee required for search and
//! insertion correctness.
use super::{
    EvaluationMode, GraphContext, HnswInvariantViolation,
    helpers::{LayerValidator, for_each_edge},
};

pub(super) fn check_bidirectional(
    ctx: GraphContext<'_>,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    let validator = LayerValidator::new(ctx.graph);
    for_each_edge(ctx.graph, |source, target, level| {
        match validator.ensure(source, target, level) {
            Ok(neighbour) => {
                if neighbour.neighbours(level).contains(&source) {
                    Ok(())
                } else {
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
