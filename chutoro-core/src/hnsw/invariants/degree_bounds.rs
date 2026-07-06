//! Node degree-bound checks for the HNSW invariants suite.
//!
//! This module exports `check_degree_bounds`, which validates each node's
//! neighbour count against the base-layer and upper-layer limits from
//! `HnswParams`. It complements `layer_consistency.rs`, which validates
//! whether those neighbours refer to valid layers, and `tests.rs`, which
//! exercises both invariant checks through shared graph fixtures.

use super::{EvaluationMode, GraphContext, HnswInvariantViolation};

pub(super) fn check_degree_bounds(
    ctx: GraphContext<'_>,
    mode: &mut EvaluationMode<'_>,
) -> Result<(), HnswInvariantViolation> {
    let upper = ctx.params.max_connections();
    let base_limit = upper
        .checked_mul(2)
        .ok_or_else(|| HnswInvariantViolation::ConfigError {
            message: format!(
                "max_connections ({upper}) overflowed when doubling for base-layer bounds"
            ),
        })?;

    for (node_id, node) in ctx.graph.nodes_iter() {
        for level in 0..node.level_count() {
            let limit = if level == 0 { base_limit } else { upper };
            let degree = node.neighbours(level).len();
            if degree > limit {
                mode.record(HnswInvariantViolation::DegreeBounds {
                    node: node_id,
                    layer: level,
                    degree,
                    limit,
                })?;
            }
        }
    }
    Ok(())
}
