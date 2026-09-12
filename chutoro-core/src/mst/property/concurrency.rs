//! Property 3: Concurrency safety.
//!
//! Runs the parallel Kruskal algorithm on the same input graph multiple
//! times and asserts that the total weight, edge count, component count,
//! and exact edge list are identical across all runs, detecting
//! non-determinism from race conditions.
//!
//! See `docs/property-testing-design.md` §4.3.3.

use proptest::test_runner::{TestCaseError, TestCaseResult};
use rayon::ThreadPoolBuilder;

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
        if !run_weight.total_cmp(&baseline_weight).is_eq() {
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

/// Runs the thread-pool determinism property for the given fixture.
///
/// Executes `parallel_kruskal` in dedicated one- and eight-thread Rayon pools
/// and asserts that both runs produce exactly the same forest.
pub(super) fn run_thread_pool_determinism_property(fixture: &MstFixture) -> TestCaseResult {
    let single_thread_pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .map_err(|error| {
            TestCaseError::fail(format!("failed to build one-thread Rayon pool: {error}"))
        })?;
    let eight_thread_pool = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .map_err(|error| {
            TestCaseError::fail(format!("failed to build eight-thread Rayon pool: {error}"))
        })?;

    let single_thread_harvest = EdgeHarvest::new(fixture.edges.clone());
    let single_thread_forest = single_thread_pool
        .install(|| parallel_kruskal(fixture.node_count, &single_thread_harvest))
        .map_err(|error| {
            TestCaseError::fail(format!(
                "one-thread parallel_kruskal failed: {error} \
                 (distribution={:?}, nodes={}, edges={})",
                fixture.distribution,
                fixture.node_count,
                fixture.edges.len(),
            ))
        })?;

    let eight_thread_harvest = EdgeHarvest::new(fixture.edges.clone());
    let eight_thread_forest = eight_thread_pool
        .install(|| parallel_kruskal(fixture.node_count, &eight_thread_harvest))
        .map_err(|error| {
            TestCaseError::fail(format!(
                "eight-thread parallel_kruskal failed: {error} \
                 (distribution={:?}, nodes={}, edges={})",
                fixture.distribution,
                fixture.node_count,
                fixture.edges.len(),
            ))
        })?;

    if single_thread_forest != eight_thread_forest {
        return Err(TestCaseError::fail(format!(
            "parallel_kruskal output diverged by Rayon thread count \
             (distribution={:?}, nodes={}, edges={}, one_thread={single_thread_forest:?}, \
             eight_threads={eight_thread_forest:?})",
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        )));
    }

    Ok(())
}
