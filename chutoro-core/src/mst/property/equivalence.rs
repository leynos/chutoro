//! Property 1: Equivalence with sequential oracle.
//!
//! For any generated input graph, verifies that the parallel Kruskal
//! implementation produces an MST with the same total weight, edge count,
//! and component count as a trusted sequential Kruskal oracle.
//!
//! See `docs/property-testing-design.md` §4.3.1.

use proptest::test_runner::{TestCaseError, TestCaseResult};

use crate::{EdgeHarvest, parallel_kruskal};

use super::oracle::sequential_kruskal;
use super::types::MstFixture;

/// Runs the oracle equivalence property for the given fixture.
///
/// Compares the parallel Kruskal output against the sequential oracle.
/// Total weights are accumulated as `f64` to avoid order-dependent `f32`
/// rounding differences.
pub(super) fn run_oracle_equivalence_property(fixture: &MstFixture) -> TestCaseResult {
    let harvest = EdgeHarvest::new(fixture.edges.clone());

    let parallel_result = parallel_kruskal(fixture.node_count, &harvest).map_err(|e| {
        TestCaseError::fail(format!(
            "parallel_kruskal failed: {e} (distribution={:?}, nodes={}, edges={})",
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        ))
    })?;

    let oracle = sequential_kruskal(fixture.node_count, &fixture.edges);

    let parallel_weight = total_weight_f64(parallel_result.edges());

    if (parallel_weight - oracle.total_weight).abs() > f64::EPSILON {
        return Err(TestCaseError::fail(format!(
            "total weight mismatch: parallel={parallel_weight}, oracle={} \
             (distribution={:?}, nodes={}, edges={})",
            oracle.total_weight,
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        )));
    }

    if parallel_result.edges().len() != oracle.edge_count {
        return Err(TestCaseError::fail(format!(
            "edge count mismatch: parallel={}, oracle={} \
             (distribution={:?}, nodes={}, edges={})",
            parallel_result.edges().len(),
            oracle.edge_count,
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        )));
    }

    if parallel_result.component_count() != oracle.component_count {
        return Err(TestCaseError::fail(format!(
            "component count mismatch: parallel={}, oracle={} \
             (distribution={:?}, nodes={}, edges={})",
            parallel_result.component_count(),
            oracle.component_count,
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        )));
    }

    Ok(())
}

/// Sums edge weights as `f64` for lossless accumulation.
fn total_weight_f64(edges: &[crate::MstEdge]) -> f64 {
    edges.iter().map(|e| f64::from(e.weight())).sum()
}
