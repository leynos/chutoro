//! Tests for parallel batch core-distance computation.
//!
//! Every case is replayed in one-thread and eight-thread Rayon pools and
//! compared with a sequential walk over the same index, so the parallel
//! searches must return the same per-index values and the same lowest-index
//! error rather than whichever result rayon observes first.

use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use crate::test_utils::{NumericTestSource, suite_proptest_config};
use crate::{CpuHnsw, DataSource, HnswParams, MetricDescriptor};

use super::super::{CoreDistanceInputs, collect_ordered_results, compute_core_distances};
use super::core_distance_support::{
    build_fixture_index, distinct_values, generated_source, sequential_core_distances, with_pool,
};

/// Rayon thread counts every result must agree on.
const THREAD_COUNTS: [usize; 2] = [1, 8];

/// Synthetic indices whose `Err` results must not change the reported error.
const FAILING_INDICES: [usize; 3] = [3, 7, 11];

/// Item count of the synthetic ordering fixture.
const SYNTHETIC_ITEMS: usize = 16;

#[rstest::rstest]
#[case::distinct_points("parity", distinct_values(24), 3)]
#[case::ties(
    "ties",
    vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    2
)]
fn parallel_searches_match_the_sequential_reference(
    #[case] name: &'static str,
    #[case] values: Vec<f32>,
    #[case] min_cluster_size: usize,
    #[values(1, 8)] threads: usize,
) {
    // Distinct metric descriptors keep each stage's distance-cache entries
    // apart, so the trial exercises concurrent misses instead of replaying
    // hits inserted by the index build or the sequential reference.
    let build_source = NumericTestSource::new(name, values.clone())
        .using_metric_descriptor(MetricDescriptor::new(format!("{name}:build")));
    let (index, ef) = build_fixture_index(&build_source).expect("fixture index must build");
    let inputs = CoreDistanceInputs::new(
        build_source.len(),
        NonZeroUsize::new(min_cluster_size).expect("case cluster sizes are non-zero"),
        ef,
    );

    let reference = NumericTestSource::new(name, values.clone())
        .using_metric_descriptor(MetricDescriptor::new(format!("{name}:reference")));
    let expected = sequential_core_distances(&reference, &index, &inputs)
        .expect("sequential reference must succeed");

    let trial = NumericTestSource::new(name, values)
        .using_metric_descriptor(MetricDescriptor::new(format!("{name}:trial")));
    let actual = with_pool(threads, || compute_core_distances(&trial, &index, &inputs))
        .expect("thread pool must build")
        .expect("parallel computation must succeed");

    assert_eq!(actual, expected, "case {name} with {threads} threads");
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn sparse_and_singleton_sources_keep_their_fallbacks(#[case] threads: usize) {
    let sparse = NumericTestSource::new("sparse", vec![0.0, 1.0]);
    let (sparse_index, sparse_ef) = build_fixture_index(&sparse).expect("fixture index must build");
    let sparse_inputs = CoreDistanceInputs::new(
        sparse.len(),
        NonZeroUsize::new(5).expect("five is non-zero"),
        sparse_ef,
    );
    let sparse_result = with_pool(threads, || {
        compute_core_distances(&sparse, &sparse_index, &sparse_inputs)
    })
    .expect("thread pool must build")
    .expect("sparse computation must succeed");
    assert_eq!(sparse_result, vec![1.0, 1.0], "thread count {threads}");

    let singleton = NumericTestSource::new("singleton", vec![0.0]);
    let (singleton_index, singleton_ef) =
        build_fixture_index(&singleton).expect("fixture index must build");
    let singleton_inputs = CoreDistanceInputs::new(
        singleton.len(),
        NonZeroUsize::new(2).expect("two is non-zero"),
        singleton_ef,
    );
    let singleton_result = with_pool(threads, || {
        compute_core_distances(&singleton, &singleton_index, &singleton_inputs)
    })
    .expect("thread pool must build")
    .expect("singleton computation must succeed");
    assert_eq!(
        singleton_result,
        vec![0.0],
        "an empty non-self list yields 0.0 with {threads} threads"
    );
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn empty_input_produces_no_results(#[case] threads: usize) {
    let source = NumericTestSource::with_len("empty", 4);
    let (index, ef) = build_fixture_index(&source).expect("fixture index must build");
    let inputs = CoreDistanceInputs::new(0, NonZeroUsize::new(2).expect("two is non-zero"), ef);

    let actual = with_pool(threads, || compute_core_distances(&source, &index, &inputs))
        .expect("thread pool must build")
        .expect("empty computation must succeed");

    assert!(actual.is_empty(), "thread count {threads}");
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn synthetic_failures_report_the_lowest_index(#[case] threads: usize) {
    // Lower indices sleep longest, so the higher-index failures finish first
    // and a completion-ordered collector would report index 11.
    let (result, completed) = with_pool(threads, || {
        let completed = AtomicUsize::new(0);
        let result = collect_ordered_results((0..SYNTHETIC_ITEMS).into_par_iter().map(|index| {
            completed.fetch_add(1, Ordering::SeqCst);
            let delay = u64::try_from(SYNTHETIC_ITEMS - index).unwrap_or(0);
            std::thread::sleep(Duration::from_millis(delay));
            if FAILING_INDICES.contains(&index) {
                Err(index)
            } else {
                Ok(index)
            }
        }));
        (result, completed.load(Ordering::SeqCst))
    })
    .expect("thread pool must build");

    assert_eq!(result, Err(3), "thread count {threads}");
    assert_eq!(
        completed, SYNTHETIC_ITEMS,
        "every index must run even after a failure (thread count {threads})"
    );
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn successful_results_keep_source_order(#[case] threads: usize) {
    let actual = with_pool(threads, || {
        collect_ordered_results((0..8usize).into_par_iter().map(|index| {
            // Reverse the completion order relative to the index order.
            std::thread::sleep(Duration::from_millis(u64::try_from(8 - index).unwrap_or(0)));
            Ok::<usize, usize>(index)
        }))
    })
    .expect("thread pool must build")
    .expect("collection must succeed");

    assert_eq!(actual, (0..8).collect::<Vec<_>>(), "thread count {threads}");
}

proptest! {
    #![proptest_config(suite_proptest_config(32))]

    #[test]
    fn parallel_searches_match_the_reference_for_generated_sources(
        (values, cluster_size) in generated_source(),
    ) {
        let build_source = NumericTestSource::new("generated", values.clone())
            .using_metric_descriptor(MetricDescriptor::new("generated:build"));
        let params = HnswParams::new(4, 8)
            .map_err(|error| TestCaseError::fail(error.to_string()))?
            .with_rng_seed(11);
        let (index, _) = CpuHnsw::build_with_edges(&build_source, params)
            .map_err(|error| TestCaseError::fail(error.to_string()))?;

        // Mirror the production `ef` formula so the generated cases search at
        // the width the stage actually requests.
        let desired = cluster_size
            .saturating_add(1)
            .max(8)
            .min(build_source.len());
        let inputs = CoreDistanceInputs::new(
            build_source.len(),
            NonZeroUsize::new(cluster_size).ok_or_else(|| {
                TestCaseError::fail("generated cluster sizes are non-zero")
            })?,
            NonZeroUsize::new(desired).unwrap_or(NonZeroUsize::MIN),
        );

        // Distinct descriptors keep the reference's and each trial's cache
        // entries apart, so every trial exercises concurrent cache misses.
        let reference = NumericTestSource::new("generated", values.clone())
            .using_metric_descriptor(MetricDescriptor::new("generated:reference"));
        let expected = sequential_core_distances(&reference, &index, &inputs)
            .map_err(|error| TestCaseError::fail(error.to_string()))?;

        for threads in THREAD_COUNTS {
            let trial = NumericTestSource::new("generated", values.clone())
                .using_metric_descriptor(MetricDescriptor::new(format!(
                    "generated:trial-{threads}"
                )));
            let actual = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| {
                    TestCaseError::fail(format!(
                        "failed to build a {threads}-thread Rayon pool: {error}"
                    ))
                })?
                .install(|| compute_core_distances(&trial, &index, &inputs))
                .map_err(|error| {
                    TestCaseError::fail(format!(
                        "parallel computation failed with {threads} threads: {error}"
                    ))
                })?;
            prop_assert_eq!(&actual, &expected, "thread count {}", threads);
        }
    }
}
