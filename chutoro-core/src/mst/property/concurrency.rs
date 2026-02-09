//! Property 3: Concurrency safety.
//!
//! Runs the parallel Kruskal algorithm on the same input graph multiple
//! times and asserts that the total weight, edge count, component count,
//! and exact edge list are identical across all runs, detecting
//! non-determinism from race conditions.
//!
//! See `docs/property-testing-design.md` §4.3.3.

use proptest::test_runner::{TestCaseError, TestCaseResult};

use crate::{EdgeHarvest, MstEdge, parallel_kruskal};

use super::helpers::total_weight_f64;
use super::types::{ConcurrencyConfig, MstFixture};

/// Runs the concurrency safety property for the given fixture.
///
/// Executes `parallel_kruskal` multiple times on the same input and
/// asserts that every run produces bit-identical results.  The repetition
/// count is controlled by [`ConcurrencyConfig`].
pub(super) fn run_concurrency_safety_property(fixture: &MstFixture) -> TestCaseResult {
    let config = ConcurrencyConfig::load();
    let harvest = EdgeHarvest::new(fixture.edges.clone());

    let baseline = parallel_kruskal(fixture.node_count, &harvest).map_err(|e| {
        TestCaseError::fail(format!(
            "baseline parallel_kruskal failed: {e} (distribution={:?}, nodes={}, edges={})",
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        ))
    })?;

    let baseline_weight = total_weight_f64(baseline.edges());
    let baseline_edges: Vec<MstEdge> = baseline.edges().to_vec();

    for run in 1..config.repetitions {
        let harvest_copy = EdgeHarvest::new(fixture.edges.clone());
        let result = parallel_kruskal(fixture.node_count, &harvest_copy).map_err(|e| {
            TestCaseError::fail(format!(
                "run {run}: parallel_kruskal failed: {e} \
                 (distribution={:?}, nodes={}, edges={})",
                fixture.distribution,
                fixture.node_count,
                fixture.edges.len(),
            ))
        })?;

        let run_weight = total_weight_f64(result.edges());
        if (run_weight - baseline_weight).abs() > f64::EPSILON {
            return Err(TestCaseError::fail(format!(
                "run {run}: total weight diverged — baseline={baseline_weight}, \
                 run={run_weight} (distribution={:?}, nodes={}, edges={})",
                fixture.distribution,
                fixture.node_count,
                fixture.edges.len(),
            )));
        }

        if result.edges().len() != baseline_edges.len() {
            return Err(TestCaseError::fail(format!(
                "run {run}: edge count diverged — baseline={}, run={} \
                 (distribution={:?})",
                baseline_edges.len(),
                result.edges().len(),
                fixture.distribution,
            )));
        }

        if result.component_count() != baseline.component_count() {
            return Err(TestCaseError::fail(format!(
                "run {run}: component count diverged — baseline={}, run={} \
                 (distribution={:?})",
                baseline.component_count(),
                result.component_count(),
                fixture.distribution,
            )));
        }

        // Exact edge-list equality — the strongest determinism check.
        if result.edges() != baseline_edges.as_slice() {
            return Err(TestCaseError::fail(format!(
                "run {run}: edge list differs from baseline \
                 (distribution={:?}, nodes={}, edges={})",
                fixture.distribution,
                fixture.node_count,
                fixture.edges.len(),
            )));
        }
    }

    Ok(())
}
