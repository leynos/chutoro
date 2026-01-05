//! Edge harvest property tests for CPU HNSW.
//!
//! Verifies that candidate edge harvesting during HNSW construction produces
//! consistent, structurally valid results suitable for MST construction.
//!
//! Properties verified:
//! 1. **Consistency**: Same input produces similar edge characteristics
//! 2. **Validity**: All edges reference valid nodes with finite distances
//! 3. **Coverage**: All inserted nodes (except entry point) appear as sources
//! 4. **No self-edges**: Source and target are always distinct
//!
//! Note: Due to Rayon's non-deterministic thread scheduling, the exact order of
//! parallel insertions can vary between runs. This affects the HNSW graph
//! structure and thus which candidate edges are discovered. We verify
//! properties that hold regardless of insertion order.

use std::collections::HashSet;

use proptest::{
    prop_assume,
    test_runner::{TestCaseError, TestCaseResult},
};

use super::graph_topology_tests::{validate_no_self_edge, validate_node_in_bounds};
use super::types::{EdgeHarvestPlan, HnswFixture};
use crate::{CpuHnsw, DataSource};

const MIN_EDGE_HARVEST_FIXTURE_LEN: usize = 2;
const MAX_EDGE_HARVEST_FIXTURE_LEN: usize = 100;

/// Validates that a distance value is finite and non-negative.
///
/// Note: This differs from graph topology's `validate_distance` which requires
/// positive distances. Edge harvest allows zero distances for coincident points.
fn validate_nonnegative_distance(distance: f32, edge_idx: usize) -> TestCaseResult {
    if !distance.is_finite() {
        return Err(TestCaseError::fail(format!(
            "edge {edge_idx}: non-finite distance {distance}"
        )));
    }
    if distance < 0.0 {
        return Err(TestCaseError::fail(format!(
            "edge {edge_idx}: negative distance {distance}"
        )));
    }
    Ok(())
}

/// Runs the edge harvest consistency property: builds an index multiple times
/// and verifies the harvested edges have similar characteristics.
///
/// Due to Rayon's non-deterministic parallel insertion order, the exact edges
/// discovered can vary between builds. We verify:
/// 1. Index sizes match exactly
/// 2. Edge counts are within reasonable tolerance
pub(super) fn run_edge_harvest_determinism_property(
    fixture: HnswFixture,
    plan: EdgeHarvestPlan,
) -> TestCaseResult {
    let params = fixture
        .params
        .build()
        .map_err(|err| TestCaseError::fail(format!("invalid params: {err}")))?;

    let source = fixture
        .clone()
        .into_source()
        .map_err(|err| TestCaseError::fail(format!("fixture -> source failed: {err}")))?;

    let len = source.len();
    prop_assume!(len >= MIN_EDGE_HARVEST_FIXTURE_LEN);
    prop_assume!(len <= MAX_EDGE_HARVEST_FIXTURE_LEN);

    // Build first time to get baseline
    let (baseline_index, baseline_edges) = CpuHnsw::build_with_edges(&source, params.clone())
        .map_err(|err| TestCaseError::fail(format!("initial build failed: {err}")))?;

    // Rebuild multiple times and compare characteristics
    for attempt in 1..=plan.rebuild_attempts() {
        let source_copy = fixture
            .clone()
            .into_source()
            .map_err(|err| TestCaseError::fail(format!("source copy failed: {err}")))?;

        let (index, edges) = CpuHnsw::build_with_edges(&source_copy, params.clone())
            .map_err(|err| TestCaseError::fail(format!("rebuild {attempt} failed: {err}")))?;

        // Index size must match exactly
        if index.len() != baseline_index.len() {
            return Err(TestCaseError::fail(format!(
                "index size mismatch on attempt {attempt}: expected {}, got {}",
                baseline_index.len(),
                index.len()
            )));
        }

        // Edge counts should be within tolerance (graph structure varies with
        // non-deterministic insertion order)
        let min_edges = baseline_edges.len().min(edges.len());
        let max_edges = baseline_edges.len().max(edges.len());
        let base_tolerance = ((min_edges as f64) * 0.3).max(3.0) as usize;
        // Tiny graphs can swing by a full neighbourhood based on insertion order.
        let tolerance = base_tolerance.max(params.max_connections());

        if max_edges > min_edges + tolerance {
            return Err(TestCaseError::fail(format!(
                "edge count divergence on attempt {attempt}: baseline {}, current {} \
                 (tolerance: {})",
                baseline_edges.len(),
                edges.len(),
                tolerance
            )));
        }
    }

    Ok(())
}

/// Runs the edge harvest validity property: verifies all harvested edges are
/// structurally valid (valid node references, finite distances, no self-edges).
pub(super) fn run_edge_harvest_validity_property(fixture: HnswFixture) -> TestCaseResult {
    let params = fixture
        .params
        .build()
        .map_err(|err| TestCaseError::fail(format!("invalid params: {err}")))?;

    let source = fixture
        .clone()
        .into_source()
        .map_err(|err| TestCaseError::fail(format!("fixture -> source failed: {err}")))?;

    let len = source.len();
    prop_assume!(len >= MIN_EDGE_HARVEST_FIXTURE_LEN);
    prop_assume!(len <= MAX_EDGE_HARVEST_FIXTURE_LEN);

    let (index, edges) = CpuHnsw::build_with_edges(&source, params)
        .map_err(|err| TestCaseError::fail(format!("build failed: {err}")))?;

    let num_nodes = index.len();

    for (i, edge) in edges.iter().enumerate() {
        validate_node_in_bounds(edge.source(), num_nodes, "source", i)?;
        validate_node_in_bounds(edge.target(), num_nodes, "target", i)?;
        validate_no_self_edge(edge.source(), edge.target(), i)?;
        validate_nonnegative_distance(edge.distance(), i)?;
    }

    Ok(())
}

/// Runs the edge harvest coverage property: verifies all inserted nodes
/// (except the entry point) appear as edge sources.
pub(super) fn run_edge_harvest_coverage_property(fixture: HnswFixture) -> TestCaseResult {
    let params = fixture
        .params
        .build()
        .map_err(|err| TestCaseError::fail(format!("invalid params: {err}")))?;

    let source = fixture
        .clone()
        .into_source()
        .map_err(|err| TestCaseError::fail(format!("fixture -> source failed: {err}")))?;

    let len = source.len();
    prop_assume!(len >= MIN_EDGE_HARVEST_FIXTURE_LEN);
    prop_assume!(len <= MAX_EDGE_HARVEST_FIXTURE_LEN);

    let (index, edges) = CpuHnsw::build_with_edges(&source, params)
        .map_err(|err| TestCaseError::fail(format!("build failed: {err}")))?;

    let num_nodes = index.len();

    // Collect all nodes that appear as edge sources
    let sources: HashSet<usize> = edges.iter().map(|e| e.source()).collect();

    // All nodes except the entry point (node 0) should appear as sources
    // Node 0 is the first inserted and has no prior nodes to discover
    for node in 1..num_nodes {
        if !sources.contains(&node) {
            return Err(TestCaseError::fail(format!(
                "node {node} not covered as edge source (num_nodes = {num_nodes})",
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::tests::property::types::{
        DistributionMetadata, HnswParamsSeed, VectorDistribution,
    };
    use rayon::ThreadPoolBuilder;
    use rstest::rstest;

    const EDGE_HARVEST_TEST_RAYON_THREADS: usize = 2; // Limit Rayon threads to reduce flakiness.

    /// Builds a dedicated Rayon thread pool with EDGE_HARVEST_TEST_RAYON_THREADS
    /// threads and runs the provided closure on it to limit edge-harvest test
    /// concurrency for improved stability. Test-only helper.
    fn with_edge_harvest_pool<T: Send>(f: impl FnOnce() -> T + Send) -> T {
        let pool = ThreadPoolBuilder::new()
            .num_threads(EDGE_HARVEST_TEST_RAYON_THREADS)
            .build()
            .expect("edge harvest test pool should build");
        pool.install(f)
    }

    fn make_fixture(vector_count: usize, seed: u64) -> HnswFixture {
        let vectors: Vec<Vec<f32>> = (0..vector_count)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32])
            .collect();
        HnswFixture {
            distribution: VectorDistribution::Uniform,
            vectors,
            metadata: DistributionMetadata::Uniform { bound: 100.0 },
            params: HnswParamsSeed {
                max_connections: 4,
                ef_construction: 16,
                level_multiplier: 1.0,
                max_level: 4,
                rng_seed: seed,
            },
        }
    }

    #[rstest]
    #[case(3, 42, 2)]
    #[case(5, 123, 3)]
    #[case(10, 456, 2)]
    #[case(20, 789, 2)]
    fn edge_harvest_determinism_rstest_cases(
        #[case] vector_count: usize,
        #[case] seed: u64,
        #[case] rebuild_attempts: usize,
    ) {
        let fixture = make_fixture(vector_count, seed);
        let plan = EdgeHarvestPlan::new(rebuild_attempts);
        with_edge_harvest_pool(|| {
            run_edge_harvest_determinism_property(fixture, plan)
                .expect("determinism property must hold");
        });
    }

    #[rstest]
    #[case(3, 42)]
    #[case(5, 123)]
    #[case(10, 456)]
    #[case(20, 789)]
    #[case(50, 999)]
    fn edge_harvest_validity_rstest_cases(#[case] vector_count: usize, #[case] seed: u64) {
        let fixture = make_fixture(vector_count, seed);
        with_edge_harvest_pool(|| {
            run_edge_harvest_validity_property(fixture).expect("validity property must hold");
        });
    }

    #[rstest]
    #[case(3, 42)]
    #[case(5, 123)]
    #[case(10, 456)]
    #[case(20, 789)]
    fn edge_harvest_coverage_rstest_cases(#[case] vector_count: usize, #[case] seed: u64) {
        let fixture = make_fixture(vector_count, seed);
        with_edge_harvest_pool(|| {
            run_edge_harvest_coverage_property(fixture).expect("coverage property must hold");
        });
    }

    #[rstest]
    fn edge_harvest_with_clustered_distribution() {
        use crate::hnsw::tests::property::types::ClusterInfo;

        // Create clustered data: two clusters
        let mut vectors = Vec::new();
        // Cluster 1: around origin
        for i in 0..5 {
            vectors.push(vec![i as f32 * 0.1, i as f32 * 0.1, 0.0]);
        }
        // Cluster 2: offset
        for i in 0..5 {
            vectors.push(vec![10.0 + i as f32 * 0.1, 10.0 + i as f32 * 0.1, 0.0]);
        }

        let fixture = HnswFixture {
            distribution: VectorDistribution::Clustered,
            vectors,
            metadata: DistributionMetadata::Clustered {
                clusters: vec![
                    ClusterInfo {
                        start: 0,
                        len: 5,
                        radius: 0.5,
                        centroid: vec![0.0, 0.0, 0.0],
                    },
                    ClusterInfo {
                        start: 5,
                        len: 5,
                        radius: 0.5,
                        centroid: vec![10.0, 10.0, 0.0],
                    },
                ],
            },
            params: HnswParamsSeed {
                max_connections: 4,
                ef_construction: 16,
                level_multiplier: 1.0,
                max_level: 4,
                rng_seed: 42,
            },
        };

        // Run all properties on clustered data
        with_edge_harvest_pool(|| {
            run_edge_harvest_validity_property(fixture.clone()).expect("validity must hold");
            run_edge_harvest_coverage_property(fixture.clone()).expect("coverage must hold");
            run_edge_harvest_determinism_property(fixture, EdgeHarvestPlan::new(2))
                .expect("determinism must hold");
        });
    }

    #[rstest]
    fn edge_harvest_two_nodes_minimal() {
        let fixture = make_fixture(2, 42);
        let plan = EdgeHarvestPlan::new(3);

        with_edge_harvest_pool(|| {
            run_edge_harvest_determinism_property(fixture.clone(), plan)
                .expect("determinism with 2 nodes");
            run_edge_harvest_validity_property(fixture.clone()).expect("validity with 2 nodes");
            run_edge_harvest_coverage_property(fixture).expect("coverage with 2 nodes");
        });
    }

    #[rstest]
    fn edge_harvest_edges_sorted_by_sequence() {
        let fixture = make_fixture(15, 77);
        let params = fixture.params.build().expect("params");
        let source = fixture.into_source().expect("source");

        let (_, edges) =
            with_edge_harvest_pool(|| CpuHnsw::build_with_edges(&source, params).expect("build"));

        // Verify edges are sorted: primary key is sequence, secondary is natural Ord.
        //
        // EdgeHarvest::from_unsorted sorts by sequence first, then by CandidateEdge's
        // Ord implementation (distance, source, target, sequence) as a tie-breaker.
        for window in edges.windows(2) {
            let (prev, curr) = (&window[0], &window[1]);
            assert!(
                prev.sequence() < curr.sequence()
                    || (prev.sequence() == curr.sequence() && prev <= curr),
                "edges must be sorted by (sequence, natural Ord): {prev:?} should come before {curr:?}",
            );
        }
    }
}
