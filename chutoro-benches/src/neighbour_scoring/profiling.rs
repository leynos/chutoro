//! Arithmetic helpers for benchmark profiling counters.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use chutoro_core::{DataSource, DataSourceError, MetricDescriptor};
use thiserror::Error;

#[derive(Debug, Error)]
pub(super) enum ProfilingError {
    #[error("build profile stats mutex poisoned")]
    StatsPoisoned,
}

#[derive(Debug, Default)]
pub(super) struct BuildProfileStats {
    pub(super) batch_calls: usize,
    pub(super) scalar_calls: usize,
    pub(super) total_batch_candidates: usize,
    pub(super) batch_scoring_time: Duration,
    pub(super) batch_sizes: Vec<usize>,
}

#[derive(Debug)]
pub(super) struct ProfilingSource<S> {
    inner: S,
    batch_calls: AtomicUsize,
    scalar_calls: AtomicUsize,
    total_batch_candidates: AtomicUsize,
    batch_scoring_nanos: AtomicU64,
    batch_sizes: Mutex<Vec<usize>>,
}

impl<S> ProfilingSource<S> {
    pub(super) const fn new(inner: S) -> Self {
        Self {
            inner,
            batch_calls: AtomicUsize::new(0),
            scalar_calls: AtomicUsize::new(0),
            total_batch_candidates: AtomicUsize::new(0),
            batch_scoring_nanos: AtomicU64::new(0),
            batch_sizes: Mutex::new(Vec::new()),
        }
    }

    pub(super) fn take_snapshot(&self) -> Result<BuildProfileStats, ProfilingError> {
        let mut batch_sizes = self
            .batch_sizes
            .lock()
            .map_err(|_| ProfilingError::StatsPoisoned)?;
        Ok(BuildProfileStats {
            batch_calls: self.batch_calls.swap(0, Ordering::Relaxed),
            scalar_calls: self.scalar_calls.swap(0, Ordering::Relaxed),
            total_batch_candidates: self.total_batch_candidates.swap(0, Ordering::Relaxed),
            batch_scoring_time: Duration::from_nanos(
                self.batch_scoring_nanos.swap(0, Ordering::Relaxed),
            ),
            batch_sizes: std::mem::take(&mut *batch_sizes),
        })
    }

    fn record_batch(&self, candidate_count: usize, elapsed: Duration) {
        if let Ok(mut batch_sizes) = self.batch_sizes.lock() {
            batch_sizes.push(candidate_count);
            saturating_add_usize(&self.batch_calls, 1);
            saturating_add_usize(&self.total_batch_candidates, candidate_count);
            saturating_add_u64(&self.batch_scoring_nanos, duration_nanos(elapsed));
        }
    }

    fn record_scalar(&self) {
        saturating_add_usize(&self.scalar_calls, 1);
    }
}

impl<S: DataSource> DataSource for ProfilingSource<S> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        self.inner.metric_descriptor()
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let result = self.inner.distance(i, j);
        self.record_scalar();
        result
    }

    fn batch_distances(
        &self,
        query: usize,
        candidates: &[usize],
    ) -> Result<Vec<f32>, DataSourceError> {
        let started = Instant::now();
        let result = self.inner.batch_distances(query, candidates);
        self.record_batch(candidates.len(), started.elapsed());
        result
    }
}

/// Converts a duration to nanoseconds, saturating at `u64::MAX`.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use chutoro_benches::neighbour_scoring::duration_nanos;
///
/// assert_eq!(duration_nanos(Duration::from_nanos(42)), 42);
/// ```
#[must_use]
pub fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Adds `amount` to an atomic `usize`, saturating at `usize::MAX`.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// use chutoro_benches::neighbour_scoring::saturating_add_usize;
///
/// let counter = AtomicUsize::new(usize::MAX);
/// saturating_add_usize(&counter, 1);
/// assert_eq!(counter.load(Ordering::Relaxed), usize::MAX);
/// ```
pub fn saturating_add_usize(target: &AtomicUsize, amount: usize) {
    let mut current = target.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_add(amount);
        match target.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

/// Adds `amount` to an atomic `u64`, saturating at `u64::MAX`.
///
/// # Examples
///
/// ```
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// use chutoro_benches::neighbour_scoring::saturating_add_u64;
///
/// let counter = AtomicU64::new(u64::MAX);
/// saturating_add_u64(&counter, 1);
/// assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
/// ```
pub fn saturating_add_u64(target: &AtomicU64, amount: u64) {
    let mut current = target.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_add(amount);
        match target.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
#[path = "profiling_source_tests.rs"]
mod source_tests;

#[cfg(test)]
mod tests {
    //! Tests for saturating profiling counters and concurrent recording.

    use std::{sync::Arc, thread, time::Duration};

    use proptest::prelude::*;

    use super::*;

    fn add_usize_repeatedly(target: &AtomicUsize, iterations: usize, amount: usize) {
        for _ in 0..iterations {
            saturating_add_usize(target, amount);
        }
    }

    fn spawn_usize_worker(
        target: Arc<AtomicUsize>,
        iterations: usize,
        amount: usize,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || add_usize_repeatedly(&target, iterations, amount))
    }

    fn add_u64_repeatedly(target: &AtomicU64, iterations: usize, amount: u64) {
        for _ in 0..iterations {
            saturating_add_u64(target, amount);
        }
    }

    fn spawn_u64_worker(
        target: Arc<AtomicU64>,
        iterations: usize,
        amount: u64,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || add_u64_repeatedly(&target, iterations, amount))
    }

    proptest! {
        #[test]
        fn duration_nanos_clamps_to_u64(
            seconds in any::<u64>(),
            subsec_nanos in 0_u32..1_000_000_000,
        ) {
            let duration = Duration::new(seconds, subsec_nanos);
            let expected = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);

            prop_assert_eq!(duration_nanos(duration), expected);
        }

        #[test]
        fn saturating_add_usize_matches_saturating_arithmetic(
            initial in any::<usize>(),
            additions in prop::collection::vec(any::<usize>(), 0..64),
        ) {
            let target = AtomicUsize::new(initial);
            let mut expected = initial;

            for amount in additions {
                expected = expected.saturating_add(amount);
                saturating_add_usize(&target, amount);
                prop_assert_eq!(target.load(Ordering::Relaxed), expected);
            }
        }

        #[test]
        fn saturating_add_u64_matches_saturating_arithmetic(
            initial in any::<u64>(),
            additions in prop::collection::vec(any::<u64>(), 0..64),
        ) {
            let target = AtomicU64::new(initial);
            let mut expected = initial;

            for amount in additions {
                expected = expected.saturating_add(amount);
                saturating_add_u64(&target, amount);
                prop_assert_eq!(target.load(Ordering::Relaxed), expected);
            }
        }
    }

    #[test]
    fn saturating_add_usize_accumulates_concurrent_updates() {
        const WORKERS: usize = 8;
        const ITERATIONS: usize = 1_024;
        const AMOUNT: usize = 3;

        let target = Arc::new(AtomicUsize::new(0));
        let handles = (0..WORKERS)
            .map(|_| {
                let worker_target = Arc::clone(&target);
                spawn_usize_worker(worker_target, ITERATIONS, AMOUNT)
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle
                .join()
                .expect("saturating_add_usize worker should not panic");
        }

        assert_eq!(
            target.load(Ordering::Relaxed),
            WORKERS * ITERATIONS * AMOUNT
        );
    }

    #[test]
    fn saturating_add_u64_accumulates_concurrent_updates() {
        const WORKERS: usize = 8;
        const ITERATIONS: usize = 1_024;
        const AMOUNT: u64 = 5;

        let target = Arc::new(AtomicU64::new(0));
        let handles = (0..WORKERS)
            .map(|_| {
                let worker_target = Arc::clone(&target);
                spawn_u64_worker(worker_target, ITERATIONS, AMOUNT)
            })
            .collect::<Vec<_>>();
        let expected_workers = u64::try_from(WORKERS).expect("worker count should fit u64");
        let expected_iterations =
            u64::try_from(ITERATIONS).expect("iteration count should fit u64");

        for handle in handles {
            handle
                .join()
                .expect("saturating_add_u64 worker should not panic");
        }

        assert_eq!(
            target.load(Ordering::Relaxed),
            expected_workers * expected_iterations * AMOUNT
        );
    }
}
