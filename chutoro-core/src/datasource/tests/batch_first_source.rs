//! Batch-first datasource test fixtures and regression tests.

use super::super::{DataSource, DataSourceError};
use rstest::{fixture, rstest};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone)]
struct BatchFirstSource {
    data: Vec<f32>,
    batch_calls: Arc<AtomicUsize>,
    distance_calls: Arc<AtomicUsize>,
}

impl BatchFirstSource {
    fn new(
        data: Vec<f32>,
        batch_calls: Arc<AtomicUsize>,
        distance_calls: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            data,
            batch_calls,
            distance_calls,
        }
    }
}

impl DataSource for BatchFirstSource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        "batch-first"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        self.distance_calls.fetch_add(1, Ordering::Relaxed);
        let l = self
            .data
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let r = self
            .data
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok((l - r).abs())
    }

    fn distance_batch(
        &self,
        pairs: &[(usize, usize)],
        out: &mut [f32],
    ) -> Result<(), DataSourceError> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }

        let mut results = Vec::with_capacity(pairs.len());
        for (left, right) in pairs.iter().copied() {
            let l = self
                .data
                .get(left)
                .ok_or(DataSourceError::OutOfBounds { index: left })?;
            let r = self
                .data
                .get(right)
                .ok_or(DataSourceError::OutOfBounds { index: right })?;
            results.push((l - r).abs());
        }
        out.copy_from_slice(&results);
        Ok(())
    }
}

fn make_batch_first(
    data: Vec<f32>,
) -> Result<(BatchFirstSource, Arc<AtomicUsize>, Arc<AtomicUsize>), DataSourceError> {
    if data.is_empty() {
        return Err(DataSourceError::EmptyData);
    }
    let batch_calls = Arc::new(AtomicUsize::new(0));
    let distance_calls = Arc::new(AtomicUsize::new(0));
    let source = BatchFirstSource::new(data, Arc::clone(&batch_calls), Arc::clone(&distance_calls));
    Ok((source, batch_calls, distance_calls))
}

#[fixture]
fn batch_first_setup()
-> Result<(BatchFirstSource, Arc<AtomicUsize>, Arc<AtomicUsize>), DataSourceError> {
    make_batch_first(vec![0.0, 1.5, 4.0])
}

#[fixture]
fn batch_first_singleton_setup()
-> Result<(BatchFirstSource, Arc<AtomicUsize>, Arc<AtomicUsize>), DataSourceError> {
    make_batch_first(vec![0.0])
}

#[rstest]
fn batch_distances_delegates_to_distance_batch(
    batch_first_setup: Result<
        (BatchFirstSource, Arc<AtomicUsize>, Arc<AtomicUsize>),
        DataSourceError,
    >,
) -> Result<(), DataSourceError> {
    let (source, batch_calls, distance_calls) = batch_first_setup?;

    let distances = source.batch_distances(0, &[1, 2])?;

    assert_eq!(distances, vec![1.5, 4.0]);
    assert_eq!(
        batch_calls.load(Ordering::Relaxed),
        1,
        "batch override should be called exactly once",
    );
    assert_eq!(
        distance_calls.load(Ordering::Relaxed),
        0,
        "scalar distance should not be used when distance_batch is available",
    );

    Ok(())
}

#[rstest]
fn batch_distances_propagates_distance_batch_errors(
    batch_first_singleton_setup: Result<
        (BatchFirstSource, Arc<AtomicUsize>, Arc<AtomicUsize>),
        DataSourceError,
    >,
) -> Result<(), DataSourceError> {
    let (source, _batch_calls, _distance_calls) = batch_first_singleton_setup?;

    let err = source
        .batch_distances(0, &[1])
        .expect_err("out-of-bounds candidate must fail");

    assert!(
        matches!(err, DataSourceError::OutOfBounds { index: 1 }),
        "unexpected error: {err:?}",
    );

    Ok(())
}

#[test]
fn distance_batch_leaves_output_unmodified_on_error() {
    let (source, _batch_calls, _distance_calls) =
        make_batch_first(vec![0.0, 1.0]).expect("fixture setup should succeed");
    let mut out = vec![10.0_f32, 20.0_f32];
    let err = source
        .distance_batch(&[(0, 1), (0, 9)], &mut out)
        .expect_err("out-of-bounds pair must fail");

    assert!(matches!(err, DataSourceError::OutOfBounds { index: 9 }));
    assert_eq!(out, vec![10.0_f32, 20.0_f32]);
}
