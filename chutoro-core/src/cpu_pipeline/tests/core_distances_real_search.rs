//! Real-search regression for the lowest-index failure contract.
//!
//! The synthetic ordering tests cover the collector in isolation. These cases
//! drive the whole HNSW search stage, so they prove that every point is
//! searched, that a failure in one search does not stop the others, and that
//! the reported error is the lowest failing source index even when a higher
//! index fails first.

use std::{
    collections::BTreeSet,
    num::NonZeroUsize,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use rstest::rstest;

use crate::test_utils::NumericTestSource;
use crate::{ChutoroError, DataSource, DataSourceError, MetricDescriptor};

use super::super::{CoreDistanceInputs, compute_core_distances};
use super::core_distance_support::{build_fixture_index, with_pool};

/// Point count of the real-search fixture.
const POINT_COUNT: usize = 12;

/// Query whose search is held back until the later failure has been observed.
const EARLY_FAILURE: usize = 3;

/// Query that fails ahead of [`EARLY_FAILURE`] when searches run concurrently.
const LATE_FAILURE: usize = 11;

/// Upper bound on waiting for the late failure.
///
/// A timeout falls through so the error assertion still runs and the ordering
/// witness fails with its own message instead of hanging the suite.
const ORDERING_WAIT: Duration = Duration::from_secs(5);

/// A source that rejects the self-pair distance for selected queries.
///
/// HNSW only needs `distance(query, query)` when it has to materialise the
/// query itself, so rejecting that pair fails exactly those queries. The source
/// also records each query it serves, and lets the early failure wait for the
/// late one, so a completion-ordered collector would report the wrong index.
struct SelfFailingSource {
    /// Backing data and metric descriptor for the trial.
    inner: NumericTestSource,
    /// Query indices whose self-pair must fail.
    failing: Vec<usize>,
    /// Query indices that reached this source, which is the left operand.
    served_queries: Mutex<BTreeSet<usize>>,
    /// Failing queries in the order their rejections were produced.
    rejections: Mutex<Vec<usize>>,
    /// Set once [`LATE_FAILURE`] has been rejected.
    late_failure_observed: AtomicBool,
    /// Set only when the early failure saw the late rejection in time.
    ordering_witnessed: AtomicBool,
    /// `Some` only when a second worker can observe the late failure.
    ordering_wait: Option<Duration>,
}

impl SelfFailingSource {
    /// Builds a source from its inner data, failing queries, and wait policy.
    const fn new(
        inner: NumericTestSource,
        failing: Vec<usize>,
        ordering_wait: Option<Duration>,
    ) -> Self {
        Self {
            inner,
            failing,
            served_queries: Mutex::new(BTreeSet::new()),
            rejections: Mutex::new(Vec::new()),
            late_failure_observed: AtomicBool::new(false),
            ordering_witnessed: AtomicBool::new(false),
            ordering_wait,
        }
    }

    /// Returns the queries this source has served, in index order.
    fn served_queries(&self) -> BTreeSet<usize> {
        // A poisoned lock still holds a coherent set: each insertion is atomic
        // under the guard, so recovering the poison cannot hide a torn write.
        self.served_queries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Returns the failing queries in the order their rejections were produced.
    fn rejections(&self) -> Vec<usize> {
        self.rejections
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Reports whether the early failure witnessed the late rejection in time.
    ///
    /// Unlike the late-failure flag itself, this is set only from inside the
    /// early failure's bounded wait. A late rejection that arrives after the
    /// deadline, or after the stage has returned, cannot set it.
    fn ordering_witnessed(&self) -> bool {
        self.ordering_witnessed.load(Ordering::SeqCst)
    }

    /// Waits, within the configured bound, for the late failure to be seen.
    ///
    /// The flag is accepted only while the deadline still holds, so a rejection
    /// that lands after the bound cannot be mistaken for an in-time witness. A
    /// timeout is not fatal here: the early query still rejects its own pair so
    /// the stage reaches its assertions, which then fail on the unset ordering
    /// witness rather than hanging the suite.
    fn await_late_failure(&self) {
        let Some(wait) = self.ordering_wait else {
            return;
        };
        let deadline = Instant::now() + wait;
        while Instant::now() < deadline {
            if self.late_failure_observed.load(Ordering::SeqCst) {
                self.ordering_witnessed.store(true, Ordering::SeqCst);
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
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
        // HNSW always passes the active query as the left operand, so this set
        // proves which searches ran. The guard is released before any wait, and
        // a poisoned lock still holds a coherent set of whole insertions.
        self.served_queries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(left);
        if left == EARLY_FAILURE && right == EARLY_FAILURE {
            self.await_late_failure();
        }
        if left == right && self.failing.contains(&left) {
            // The rejection is logged before the wake-up flag is set, so the
            // recorded order cannot show the early failure first merely because
            // it woke up faster than the late failure finished logging.
            self.rejections
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(left);
            if left == LATE_FAILURE {
                self.late_failure_observed.store(true, Ordering::SeqCst);
            }
            // A synthetic rejection; the index is preserved in the error so
            // the test can prove which query produced it.
            return Err(DataSourceError::OutOfBounds { index: left });
        }
        self.inner.distance(left, right)
    }
}

#[rstest]
#[case::single_thread(1)]
#[case::eight_threads(8)]
fn real_search_failure_reports_the_lowest_failing_query(#[case] threads: usize) {
    let data: Vec<f32> = (0..POINT_COUNT)
        .map(|point| f32::from(u8::try_from(point).expect("point indices fit in u8")))
        .collect();
    let build_source = NumericTestSource::new("self-fail", data.clone())
        .using_metric_descriptor(MetricDescriptor::new("self-fail:build"));
    let (index, ef) = build_fixture_index(&build_source).expect("fixture index must build");
    let inputs = CoreDistanceInputs::new(
        build_source.len(),
        NonZeroUsize::new(2).expect("two is non-zero"),
        ef,
    );

    // A descriptor per thread count keeps the clean trial off the index build's
    // cache entries, so both trials run concurrent searches on a cold cache.
    let clean_source = NumericTestSource::new("self-fail", data.clone())
        .using_metric_descriptor(MetricDescriptor::new(format!("self-fail:clean-{threads}")));
    let clean = with_pool(threads, || {
        compute_core_distances(&clean_source, &index, &inputs)
    })
    .expect("thread pool must build")
    .expect("clean trial must succeed");
    assert_eq!(clean.len(), POINT_COUNT, "thread count {threads}");

    // Distinct metric descriptors give each trial a fresh distance cache, so
    // neither trial can be answered from the other trial's cache entries.
    for trial in ["a", "b"] {
        let descriptor = format!("self-fail:trial-{threads}-{trial}");
        // Only a pool with a second worker may wait; a one-thread pool would
        // wait for a failure it is itself holding back.
        let ordering_wait = (threads > 1).then_some(ORDERING_WAIT);
        let failing_source = SelfFailingSource::new(
            NumericTestSource::new("self-fail", data.clone())
                .using_metric_descriptor(MetricDescriptor::new(descriptor.clone())),
            vec![EARLY_FAILURE, LATE_FAILURE],
            ordering_wait,
        );

        let error = with_pool(threads, || {
            compute_core_distances(&failing_source, &index, &inputs)
        })
        .expect("thread pool must build")
        .expect_err("a rejected self-pair must fail the stage");

        assert!(
            matches!(
                error,
                ChutoroError::DataSource {
                    error: DataSourceError::OutOfBounds {
                        index: EARLY_FAILURE
                    },
                    ..
                }
            ),
            "trial {descriptor} with {threads} threads must report the lowest failing query, \
             got {error:?}"
        );
        assert_eq!(
            failing_source.served_queries(),
            (0..POINT_COUNT).collect::<BTreeSet<_>>(),
            "trial {descriptor} with {threads} threads must search every point even after a failure"
        );
        if threads > 1 {
            assert!(
                failing_source.ordering_witnessed(),
                "trial {descriptor} with {threads} threads timed out waiting for query \
                 {LATE_FAILURE}, so the lowest-index ordering was never witnessed"
            );
            assert_eq!(
                failing_source.rejections(),
                vec![LATE_FAILURE, EARLY_FAILURE],
                "trial {descriptor} with {threads} threads must reject query {LATE_FAILURE} \
                 before query {EARLY_FAILURE}"
            );
        }
    }
}
