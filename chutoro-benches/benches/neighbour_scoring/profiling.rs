//! Profiling wrapper for HNSW build-time neighbour-scoring diagnostics.

use std::{
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use chutoro_benches::neighbour_scoring::{
    duration_nanos, saturating_add_u64, saturating_add_usize,
};
use chutoro_core::{DataSource, DataSourceError, MetricDescriptor};

use super::support::{BenchError, BenchResult};

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

    pub(super) fn take_snapshot(&self) -> BenchResult<BuildProfileStats> {
        let mut batch_sizes = self
            .batch_sizes
            .lock()
            .map_err(|_| BenchError::BuildProfileStatsPoisoned)?;
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
        let elapsed = started.elapsed();
        self.record_batch(candidates.len(), elapsed);
        result
    }
}
