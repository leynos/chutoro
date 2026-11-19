//! Unit tests for the batch-aware peak-memory estimate.

use rstest::rstest;

use super::*;
use crate::memory::estimate_peak_bytes_for_hnsw_params;

// -- estimate_peak_bytes: concurrency -----------------------------------

/// Cluster size used by the concurrency cases.
const CLUSTER: std::num::NonZeroUsize = std::num::NonZeroUsize::MIN;

#[cfg(feature = "cpu")]
#[rstest]
#[case::single_search(1)]
#[case::eight_workers(8)]
#[case::many_workers(64)]
fn concurrency_estimate_keeps_one_search_state_per_worker(#[case] concurrent_searches: usize) {
    use std::num::NonZeroUsize;

    let params = crate::HnswParams::new(4, 16).expect("parameters must be valid");
    let searches = NonZeroUsize::new(concurrent_searches).expect("case counts are non-zero");
    let with_concurrency =
        estimate_peak_bytes_for_hnsw_params_with_concurrency(100, CLUSTER, &params, searches);

    if concurrent_searches == 1 {
        assert_eq!(
            with_concurrency,
            estimate_peak_bytes_for_hnsw_params(100, &params),
            "a single concurrent search must match the one-state estimate"
        );
    } else {
        assert!(
            with_concurrency > estimate_peak_bytes_for_hnsw_params(100, &params),
            "{concurrent_searches} concurrent searches must raise the estimate"
        );
    }
}

#[cfg(feature = "cpu")]
#[test]
fn concurrency_estimate_scales_with_worker_count() {
    use std::num::NonZeroUsize;

    let params = crate::HnswParams::new(4, 64).expect("parameters must be valid");
    let two = estimate_peak_bytes_for_hnsw_params_with_concurrency(
        100,
        CLUSTER,
        &params,
        NonZeroUsize::new(2).expect("two is non-zero"),
    );
    let four = estimate_peak_bytes_for_hnsw_params_with_concurrency(
        100,
        CLUSTER,
        &params,
        NonZeroUsize::new(4).expect("four is non-zero"),
    );

    assert!(
        four > two,
        "doubling the concurrent search count must raise the estimate"
    );
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::below_construction_width(2, 16)]
#[case::above_construction_width(64, 65)]
#[case::wider_than_the_dataset(500, 100)]
fn batch_search_width_follows_the_cluster_size(
    #[case] min_cluster_size: usize,
    #[case] expected: usize,
) {
    let params = crate::HnswParams::new(4, 16).expect("parameters must be valid");
    let cluster = std::num::NonZeroUsize::new(min_cluster_size).expect("case sizes are non-zero");

    assert_eq!(
        batch_search_width(100, cluster, &params).get(),
        expected,
        "the batch width must combine the cluster size, the construction width, and the dataset"
    );
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::single_worker(1)]
#[case::eight_workers(8)]
fn a_large_cluster_size_raises_the_concurrent_search_term(#[case] concurrent_searches: usize) {
    let params = crate::HnswParams::new(4, 16).expect("parameters must be valid");
    let searches =
        std::num::NonZeroUsize::new(concurrent_searches).expect("case counts are non-zero");
    let small_cluster = std::num::NonZeroUsize::new(2).expect("two is non-zero");
    let large_cluster = std::num::NonZeroUsize::new(1_000).expect("one thousand is non-zero");

    let small = estimate_peak_bytes_for_hnsw_params_with_concurrency(
        2_000,
        small_cluster,
        &params,
        searches,
    );
    let large = estimate_peak_bytes_for_hnsw_params_with_concurrency(
        2_000,
        large_cluster,
        &params,
        searches,
    );

    assert!(
        large > small,
        "with {concurrent_searches} concurrent searches, a cluster size wider than the \
         construction breadth must raise the estimate"
    );
}
