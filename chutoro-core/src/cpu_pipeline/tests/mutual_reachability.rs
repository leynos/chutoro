//! Property tests for the parallel mutual-reachability transform.
//!
//! The generated edge lists draw endpoints and sequence numbers from small
//! ranges, so duplicate edges and shared sequences appear naturally next to
//! disconnected components and varied weights. Each case is replayed in
//! one-thread and eight-thread Rayon pools and compared with the sequential
//! reference, which pins both the transform's result and its determinism.

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use rayon::ThreadPoolBuilder;

use crate::mst::{parallel_kruskal, parallel_kruskal_from_edges};
use crate::test_utils::suite_proptest_config;
use crate::{CandidateEdge, EdgeHarvest};

use super::super::mutual_reachability_edges;
use super::reference::sequential_mutual_edges;

/// Rayon thread counts the transform must agree on.
const THREAD_COUNTS: [usize; 2] = [1, 8];

proptest! {
    #![proptest_config(suite_proptest_config(64))]

    #[test]
    fn parallel_transform_matches_the_sequential_form(
        (edges, core_distances) in fixture_strategy(),
    ) {
        let harvest = EdgeHarvest::new(edges);
        let expected = sequential_mutual_edges(&harvest, &core_distances).ok_or_else(|| {
            TestCaseError::fail("every generated endpoint must have a core distance")
        })?;

        for threads in THREAD_COUNTS {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| {
                    TestCaseError::fail(format!(
                        "failed to build a {threads}-thread Rayon pool: {error}"
                    ))
                })?;
            let actual = pool
                .install(|| mutual_reachability_edges("fixture", &harvest, &core_distances))
                .map_err(|error| {
                    TestCaseError::fail(format!(
                        "transform failed with {threads} threads: {error}"
                    ))
                })?;
            prop_assert_eq!(&actual, &expected, "thread count {}", threads);
        }
    }

    /// Compare a whole forest built through the direct slice path with one
    /// built through the sequential transform and `EdgeHarvest::new`.
    ///
    /// The transform-only property above pins the edge weights; this one
    /// pins the downstream forest, so a difference that only appears after
    /// canonicalisation, sorting, or deduplication cannot slip through.
    #[test]
    fn direct_slice_forest_matches_the_harvest_path(
        (edges, core_distances) in fixture_strategy(),
    ) {
        let node_count = core_distances.len();
        let harvest = EdgeHarvest::new(edges);
        let sequential = sequential_mutual_edges(&harvest, &core_distances).ok_or_else(|| {
            TestCaseError::fail("every generated endpoint must have a core distance")
        })?;
        let harvest_path = EdgeHarvest::new(sequential);

        for threads in THREAD_COUNTS {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| {
                    TestCaseError::fail(format!(
                        "failed to build a {threads}-thread Rayon pool: {error}"
                    ))
                })?;
            let (direct, via_harvest) = pool
                .install(|| -> core::result::Result<_, String> {
                    let weighted = mutual_reachability_edges("fixture", &harvest, &core_distances)
                        .map_err(|error| error.to_string())?;
                    let direct = parallel_kruskal_from_edges(node_count, &weighted)
                        .map_err(|error| error.to_string())?;
                    let via_harvest = parallel_kruskal(node_count, &harvest_path)
                        .map_err(|error| error.to_string())?;
                    Ok((direct, via_harvest))
                })
                .map_err(|error| {
                    TestCaseError::fail(format!(
                        "forest construction failed with {threads} threads: {error}"
                    ))
                })?;

            prop_assert_eq!(
                direct.edges(),
                via_harvest.edges(),
                "forest edges must match with {} threads",
                threads
            );
            prop_assert_eq!(
                direct.component_count(),
                via_harvest.component_count(),
                "component counts must match with {} threads",
                threads
            );
        }
    }
}

/// Generates a candidate-edge list and the matching core distances.
fn fixture_strategy() -> impl Strategy<Value = (Vec<CandidateEdge>, Vec<f32>)> {
    (1usize..10).prop_flat_map(|node_count| {
        (
            prop::collection::vec(edge_strategy(node_count), 0..32),
            prop::collection::vec(0.0f32..8.0, node_count),
        )
    })
}

/// Generates one candidate edge whose endpoints fit the core-distance vector.
fn edge_strategy(node_count: usize) -> impl Strategy<Value = CandidateEdge> {
    (0..node_count, 0..node_count, 0.0f32..8.0, 0u64..4).prop_map(
        |(source, target, distance, sequence)| {
            CandidateEdge::new(source, target, distance, sequence)
        },
    )
}
