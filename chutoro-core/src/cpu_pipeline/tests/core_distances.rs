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
use crate::{
    ChutoroError, CpuHnsw, DataSource, DataSourceError, HnswParams, MetricDescriptor, Neighbour,
};

use super::super::{CoreDistanceInputs, collect_ordered_results, compute_core_distances};

/// Rayon thread counts every result must agree on.
const THREAD_COUNTS: [usize; 2] = [1, 8];

/// Synthetic indices whose `Err` results must not change the reported error.
const FAILING_INDICES: [usize; 3] = [3, 7, 11];

/// Item count of the synthetic ordering fixture.
const SYNTHETIC_ITEMS: usize = 16;

/// Runs `task` inside a Rayon pool with exactly `threads` workers.
///
/// Building the pool can fail only for an unusable thread count or an operating
/// system resource failure, so callers decide how to report that.
fn with_pool<T: Send>(
    threads: usize,
    task: impl FnOnce() -> T + Send,
) -> Result<T, rayon::ThreadPoolBuildError> {
    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    Ok(pool.install(task))
}

/// Builds a deterministic index over `source` and the `ef` its searches use.
///
/// The `ef` mirrors the production formula for this stage so the tests
/// exercise the same search width the pipeline requests.
fn build_fixture_index(
    source: &NumericTestSource,
) -> Result<(CpuHnsw, NonZeroUsize), crate::HnswError> {
    let params = HnswParams::new(4, 16)?.with_rng_seed(42);
    let (index, _) = CpuHnsw::build_with_edges(source, params)?;
    // `max(1)` keeps the search width non-zero for an empty source.
    let ef = NonZeroUsize::new(source.len().max(1)).unwrap_or(NonZeroUsize::MIN);
    Ok((index, ef))
}

/// Computes core distances serially, restating the selection rule directly.
///
/// This deliberately does not call the shared helper: it is the independent
/// reference the parallel path is compared against.
fn sequential_core_distances<D: DataSource + Sync>(
    source: &D,
    index: &CpuHnsw,
    inputs: &CoreDistanceInputs,
) -> Result<Vec<f32>, crate::HnswError> {
    let mut core_distances = Vec::with_capacity(inputs.items);
    for point in 0..inputs.items {
        let neighbours = index.search(source, point, inputs.ef)?;
        let others: Vec<Neighbour> = neighbours
            .into_iter()
            .filter(|neighbour| neighbour.id != point)
            .collect();
        let core = if others.len() >= inputs.min_cluster_size.get() {
            others
                .get(inputs.min_cluster_size.get().saturating_sub(1))
                .map_or(0.0, |neighbour| neighbour.distance)
        } else {
            others.last().map_or(0.0, |neighbour| neighbour.distance)
        };
        core_distances.push(core);
    }
    Ok(core_distances)
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn parallel_searches_match_the_sequential_reference(#[case] threads: usize) {
    // Distinct metric descriptors keep the stages' distance-cache entries
    // apart, so the trials exercise concurrent misses instead of replaying
    // hits inserted by the index build or the sequential reference.
    let build_source = NumericTestSource::with_len("parity", 24)
        .using_metric_descriptor(MetricDescriptor::new("parity:build"));
    let (index, ef) = build_fixture_index(&build_source).expect("fixture index must build");
    let inputs = CoreDistanceInputs::new(
        build_source.len(),
        NonZeroUsize::new(3).expect("three is non-zero"),
        ef,
    );

    let reference = NumericTestSource::with_len("parity", 24)
        .using_metric_descriptor(MetricDescriptor::new("parity:reference"));
    let expected = sequential_core_distances(&reference, &index, &inputs)
        .expect("sequential reference must succeed");

    let trial = NumericTestSource::with_len("parity", 24)
        .using_metric_descriptor(MetricDescriptor::new("parity:trial"));
    let actual = with_pool(threads, || compute_core_distances(&trial, &index, &inputs))
        .expect("thread pool must build")
        .expect("parallel computation must succeed");

    assert_eq!(actual, expected, "thread count {threads}");
}

#[rstest::rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn ties_resolve_identically_at_every_thread_count(#[case] threads: usize) {
    // Pairs of equal points make every distance a tie between two ids.
    let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    let build_source = NumericTestSource::new("ties", data.clone())
        .using_metric_descriptor(MetricDescriptor::new("ties:build"));
    let (index, ef) = build_fixture_index(&build_source).expect("fixture index must build");
    let inputs = CoreDistanceInputs::new(
        build_source.len(),
        NonZeroUsize::new(2).expect("two is non-zero"),
        ef,
    );

    // As in the parity test, distinct descriptors give each stage its own
    // cache entries so the parallel trial starts cold.
    let reference = NumericTestSource::new("ties", data.clone())
        .using_metric_descriptor(MetricDescriptor::new("ties:reference"));
    let expected = sequential_core_distances(&reference, &index, &inputs)
        .expect("sequential reference must succeed");

    let trial = NumericTestSource::new("ties", data)
        .using_metric_descriptor(MetricDescriptor::new("ties:trial"));
    let actual = with_pool(threads, || compute_core_distances(&trial, &index, &inputs))
        .expect("thread pool must build")
        .expect("parallel computation must succeed");

    assert_eq!(actual, expected, "thread count {threads}");
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

/// A source that rejects the self-pair distance for selected queries.
///
/// HNSW only needs `distance(query, query)` when it has to materialise the
/// query itself, so rejecting that pair fails exactly those queries.
struct SelfFailingSource {
    /// Backing data and metric descriptor for the trial.
    inner: NumericTestSource,
    /// Query indices whose self-pair must fail.
    failing: Vec<usize>,
}

impl DataSource for SelfFailingSource {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        self.inner.metric_descriptor()
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        if left == right && self.failing.contains(&left) {
            // A synthetic rejection; the index is preserved in the error so
            // the test can prove which query produced it.
            return Err(DataSourceError::OutOfBounds { index: left });
        }
        self.inner.distance(left, right)
    }
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    reason = "the fixture is a handwritten sequence of small whole numbers"
)]
fn real_search_failure_reports_the_lowest_failing_query() {
    let data: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let build_source = NumericTestSource::new("self-fail", data.clone())
        .using_metric_descriptor(MetricDescriptor::new("self-fail:build"));
    let (index, ef) = build_fixture_index(&build_source).expect("fixture index must build");
    let inputs = CoreDistanceInputs::new(
        build_source.len(),
        NonZeroUsize::new(2).expect("two is non-zero"),
        ef,
    );

    let clean =
        compute_core_distances(&build_source, &index, &inputs).expect("clean trial must succeed");
    assert_eq!(clean.len(), 12);

    // Distinct metric descriptors give each trial a fresh distance cache, so
    // neither trial can be answered from the other trial's cache entries.
    for descriptor in ["self-fail:trial-a", "self-fail:trial-b"] {
        let failing_source = SelfFailingSource {
            inner: NumericTestSource::new("self-fail", data.clone())
                .using_metric_descriptor(MetricDescriptor::new(descriptor)),
            failing: vec![3, 11],
        };

        let error = compute_core_distances(&failing_source, &index, &inputs)
            .expect_err("a rejected self-pair must fail the stage");

        assert!(
            matches!(
                error,
                ChutoroError::DataSource {
                    error: DataSourceError::OutOfBounds { index: 3 },
                    ..
                }
            ),
            "trial {descriptor} must report the lowest failing query, got {error:?}"
        );
    }
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

/// Generates a source's values and a plausible minimum cluster size.
fn generated_source() -> impl Strategy<Value = (Vec<f32>, usize)> {
    (2usize..10)
        .prop_flat_map(|point_count| (prop::collection::vec(0.0f32..16.0, point_count), 1usize..6))
}
