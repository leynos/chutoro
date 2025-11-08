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
        let neighbour = match validator.ensure(source, target, level) {
            Ok(node) => node,
            Err(err) => return mode.record(err),
        };
        if neighbour.neighbours(level).contains(&source) {
            return Ok(());
        }
        mode.record(HnswInvariantViolation::MissingBacklink {
            origin: source,
            target,
            layer: level,
        })?;
        Ok(())
    })
}
