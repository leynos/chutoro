//! Property tests for the parallel mutual-reachability transform.
//!
//! The generated edge lists draw endpoints and sequence numbers from small
//! ranges, so duplicate edges and shared sequences appear naturally next to
//! disconnected components and varied weights. Each case is replayed in
//! one-thread and eight-thread Rayon pools and compared with the sequential
//! reference, which pins both the transform's result and its determinism.
//! A generated property and example-based cases then pin which lookup error
//! the transform reports when several harvested endpoints have no core
//! distance, and in which order the two endpoint roles are checked.

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use rayon::ThreadPoolBuilder;

use crate::mst::{parallel_kruskal, parallel_kruskal_from_edges};
use crate::test_utils::suite_proptest_config;
use crate::{CandidateEdge, EdgeHarvest};

use super::super::mutual_reachability_edges;
use super::core_distance_support::with_pool;
use super::reference::sequential_mutual_edges;

/// Rayon thread counts the transform must agree on.
const THREAD_COUNTS: [usize; 2] = [1, 8];

/// Harvest length for the lookup-error cases; large enough to split across
/// every worker of an eight-thread pool.
const LOOKUP_HARVEST_LEN: usize = 4096;

/// Number of nodes that have a core distance in the lookup-error cases.
const LOOKUP_NODE_COUNT: usize = 4;

/// Last index of the harvest's first half. With the next index it straddles
/// the midpoint, where rayon's first split puts the two on different workers.
const LOW_FAILURE_INDEX: usize = 2047;

/// First index of the harvest's second half.
const HIGH_FAILURE_INDEX: usize = 2048;

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

    /// Pins the lowest-index lookup rule on generated inputs.
    ///
    /// The transform takes a plain `&[f32]`, so an absent core distance can
    /// only be an index at or beyond the slice's end: the failing endpoints
    /// form a suffix, not an arbitrary subset. The oracle therefore walks the
    /// harvest in input order and checks each edge's source before its target,
    /// which is the contract stated directly. Both pinned pools must name the
    /// endpoint that oracle names, in the same role.
    #[test]
    fn missing_core_distances_report_the_lowest_index_property(
        (edges, core_len) in lookup_error_strategy(),
    ) {
        let harvest = EdgeHarvest::new(edges);
        let core_distances = vec![0.25_f32; core_len];
        let expected = first_missing_endpoint(harvest.as_slice(), core_len).ok_or_else(|| {
            TestCaseError::fail("strategy must leave an endpoint without a core distance")
        })?;

        for threads in THREAD_COUNTS {
            let message = with_pool(threads, || {
                mutual_reachability_edges("fixture", &harvest, &core_distances)
            })
            .map_err(|error| {
                TestCaseError::fail(format!(
                    "failed to build a {threads}-thread Rayon pool: {error}"
                ))
            })?
            .err()
            .map(|error| error.to_string())
            .ok_or_else(|| {
                TestCaseError::fail(format!(
                    "transform must reject the input with {threads} threads"
                ))
            })?;

            prop_assert!(
                message.contains(&format!("harvested edge {expected} has no core distance")),
                "thread count {} reported {:?}, expected the {} lookup",
                threads,
                message,
                expected,
            );
        }
    }
}

/// Generates a harvest with a non-empty suffix of nodes lacking a core
/// distance, together with the length of the core-distance vector.
///
/// An edge to the last node is appended, and `core_len` is always below the
/// node count, so the harvest cannot accidentally avoid every missing node.
/// The small generated list is then tiled up to `LOOKUP_HARVEST_LEN`, so a
/// collector that reports whichever failure a worker reached first has
/// several failing indices spread across the workers to choose between.
fn lookup_error_strategy() -> impl Strategy<Value = (Vec<CandidateEdge>, usize)> {
    (2usize..8).prop_flat_map(|node_count| {
        (
            prop::collection::vec(edge_strategy(node_count), 1..8),
            0..node_count,
        )
            .prop_map(move |(mut edges, core_len)| {
                edges.push(CandidateEdge::new(0, node_count - 1, 0.5, 0));
                let pattern = edges.clone();
                while edges.len() < LOOKUP_HARVEST_LEN {
                    edges.extend(pattern.iter().copied());
                }
                (edges, core_len)
            })
    })
}

/// Returns the `"source N"` or `"target N"` the transform must name, or
/// `None` when every endpoint has a core distance.
fn first_missing_endpoint(edges: &[CandidateEdge], core_len: usize) -> Option<String> {
    edges.iter().find_map(|edge| {
        if edge.source() >= core_len {
            Some(format!("source {}", edge.source()))
        } else if edge.target() >= core_len {
            Some(format!("target {}", edge.target()))
        } else {
            None
        }
    })
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

/// Builds a valid harvest whose edge at each listed index is replaced.
///
/// Sequence numbers equal input indices, so the harvest's sequence order is
/// the input order the transform walks.
/// Valid edges join nodes 0 and 1; every node has a core distance.
fn harvest_with_replacements(
    replacements: &[(usize, usize, usize)],
) -> Result<EdgeHarvest, std::num::TryFromIntError> {
    let edges = (0..LOOKUP_HARVEST_LEN)
        .map(|index| {
            let (source, target) = replacements
                .iter()
                .find(|(at, _, _)| *at == index)
                .map_or((0, 1), |(_, s, t)| (*s, *t));
            let sequence = u64::try_from(index)?;
            Ok(CandidateEdge::new(source, target, 0.5, sequence))
        })
        .collect::<Result<Vec<CandidateEdge>, std::num::TryFromIntError>>()?;
    Ok(EdgeHarvest::new(edges))
}

/// Runs the transform in a pool of `threads` workers and returns its error
/// text, or `None` when the transform succeeded.
fn transform_error(
    threads: usize,
    harvest: &EdgeHarvest,
) -> Result<Option<String>, rayon::ThreadPoolBuildError> {
    let core_distances = vec![0.25; LOOKUP_NODE_COUNT];
    let result = with_pool(threads, || {
        mutual_reachability_edges("fixture", harvest, &core_distances)
    })?;
    Ok(result.err().map(|error| error.to_string()))
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn missing_core_distances_report_the_lowest_index(#[case] threads: usize) {
    // A completion-ordered collector can report the higher index, because
    // the two failures run on different workers.
    let harvest =
        harvest_with_replacements(&[(LOW_FAILURE_INDEX, 9, 0), (HIGH_FAILURE_INDEX, 0, 7)])
            .expect("fixture indices fit in u64");

    for _ in 0..32 {
        let message = transform_error(threads, &harvest).expect("thread pool must build");
        assert!(
            message
                .as_deref()
                .is_some_and(|text| text.contains("source 9")),
            "thread count {threads} reported {message:?}"
        );
    }
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn a_missing_source_is_reported_before_a_missing_target(#[case] threads: usize) {
    let harvest = harvest_with_replacements(&[(0, 9, 8)]).expect("fixture indices fit in u64");

    let message = transform_error(threads, &harvest).expect("thread pool must build");

    assert!(
        message
            .as_deref()
            .is_some_and(|text| text.contains("source 9")),
        "thread count {threads} reported {message:?}"
    );
}
