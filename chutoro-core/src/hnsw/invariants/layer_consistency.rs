use super::helpers::for_each_edge;
use super::{EvaluationMode, GraphContext, HnswInvariantViolation, helpers::LayerValidator};

pub(super) fn check_layer_consistency(
    ctx: GraphContext<'_>,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    let validator = LayerValidator::new(ctx.graph);
    for_each_edge(ctx.graph, |source, target, level| {
        match validator.ensure(source, target, level) {
            Ok(_) => Ok(()),
            Err(err) => mode.record(err),
        }
    })?;

    if let Some(entry) = ctx.graph.entry() {
        if let Err(err) = validator.ensure(entry.node, entry.node, entry.level) {
            mode.record(err)?;
        }
    }
    Ok(())
}
