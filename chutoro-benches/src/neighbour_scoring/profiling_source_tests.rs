//! Tests for the neighbour-scoring profiling source.

use std::{thread, time::Duration};

use chutoro_core::{DataSource, DataSourceError, MetricDescriptor};

use super::ProfilingSource;

#[derive(Debug)]
struct StubSource {
    len: usize,
    distances_from_zero: Vec<f32>,
    batch_delay: Duration,
}

impl DataSource for StubSource {
    fn len(&self) -> usize {
        self.len
    }

    fn name(&self) -> &'static str {
        "stub"
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("stub:absolute")
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        if i >= self.len {
            return Err(DataSourceError::OutOfBounds { index: i });
        }
        self.distances_from_zero
            .get(j)
            .copied()
            .ok_or(DataSourceError::OutOfBounds { index: j })
    }

    fn batch_distances(
        &self,
        query: usize,
        candidates: &[usize],
    ) -> Result<Vec<f32>, DataSourceError> {
        thread::sleep(self.batch_delay);
        candidates
            .iter()
            .map(|&candidate| self.distance(query, candidate))
            .collect()
    }
}

fn source() -> ProfilingSource<StubSource> {
    ProfilingSource::new(StubSource {
        len: 4,
        distances_from_zero: vec![0.0, 1.5, 3.0, 7.0],
        batch_delay: Duration::from_millis(1),
    })
}

#[test]
fn profiling_source_initializes_counters_to_zero() {
    let source = source();
    let stats = source.take_snapshot().expect("snapshot must be available");

    assert_eq!(stats.batch_calls, 0);
    assert_eq!(stats.scalar_calls, 0);
    assert_eq!(stats.total_batch_candidates, 0);
    assert!(stats.batch_scoring_time.is_zero());
    assert!(stats.batch_sizes.is_empty());
}

#[test]
fn profiling_source_records_scalar_calls() {
    let source = source();

    let distance = source.distance(0, 2).expect("distance must succeed");
    assert_eq!(distance.to_bits(), 3.0_f32.to_bits());
    let stats = source.take_snapshot().expect("snapshot must be available");

    assert_eq!(stats.scalar_calls, 1);
    assert_eq!(stats.batch_calls, 0);
}

#[test]
fn profiling_source_records_batches_and_resets_snapshots() {
    let source = source();

    let distances = source
        .batch_distances(0, &[1, 2, 3])
        .expect("batch distances must succeed");
    assert_eq!(distances, vec![1.5, 3.0, 7.0]);

    let stats = source.take_snapshot().expect("snapshot must be available");
    assert_eq!(stats.batch_calls, 1);
    assert_eq!(stats.scalar_calls, 0);
    assert_eq!(stats.total_batch_candidates, 3);
    assert!(stats.batch_scoring_time >= Duration::from_millis(1));
    assert_eq!(stats.batch_sizes, vec![3]);

    let reset = source.take_snapshot().expect("snapshot must be available");
    assert_eq!(reset.batch_calls, 0);
    assert_eq!(reset.scalar_calls, 0);
    assert_eq!(reset.total_batch_candidates, 0);
    assert!(reset.batch_scoring_time.is_zero());
    assert!(reset.batch_sizes.is_empty());
}
