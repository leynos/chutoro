//! Integration tests for the DataSource trait behaviour.

mod common;

use chutoro_core::{DataSource, DataSourceError};
use common::Dummy;
use rstest::{fixture, rstest};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[fixture]
fn dummy(#[default(vec![1.0, 2.0])] data: Vec<f32>) -> Dummy {
    Dummy::new(data)
}

#[rstest(dummy(vec![1.0, 3.0, 6.0]))]
fn distance_batch_returns_distances(dummy: Dummy) {
    let pairs = [(0, 1), (1, 2)];
    let mut out = [0.0; 2];
    dummy
        .distance_batch(&pairs, &mut out)
        .expect("distance_batch must succeed");
    assert_eq!(out, [2.0, 3.0]);
}

#[rstest]
#[case::short_output(vec![(0, 1)], vec![])]
#[case::long_output(vec![(0, 1)], vec![0.0, 0.0])]
fn distance_batch_length_mismatch(
    dummy: Dummy,
    #[case] pairs: Vec<(usize, usize)>,
    #[case] mut out: Vec<f32>,
) {
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .expect_err("distance_batch must report length mismatch");
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
}

#[rstest]
fn distance_out_of_bounds(dummy: Dummy) {
    let err = dummy
        .distance(0, 2)
        .expect_err("distance must check bounds");
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 2 }));
}

#[rstest]
fn distance_batch_preserves_out_on_error(dummy: Dummy) {
    let pairs = [(0, 1), (0, 99)];
    let mut out = [1.0_f32, 1.0];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .expect_err("distance_batch must propagate inner errors");
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 99 }));
    assert_eq!(out, [1.0, 1.0]);
}

#[rstest(dummy(vec![]))]
fn distance_batch_empty_ok(dummy: Dummy) {
    let pairs: [(usize, usize); 0] = [];
    let mut out: [f32; 0] = [];
    assert!(dummy.distance_batch(&pairs, &mut out).is_ok());
}

#[rstest(dummy(vec![1.0]))]
fn distance_batch_empty_pairs_error(dummy: Dummy) {
    let pairs: [(usize, usize); 0] = [];
    let mut out = [0.0];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .expect_err("distance_batch must report length mismatch");
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
}

#[rstest]
fn distance_batch_empty_output_error(dummy: Dummy) {
    let pairs = [(0, 1)];
    let mut out: [f32; 0] = [];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .expect_err("distance_batch must report length mismatch");
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
}

#[derive(Clone)]
struct BatchFirstDummy {
    data: Vec<f32>,
    batch_calls: Arc<AtomicUsize>,
    distance_calls: Arc<AtomicUsize>,
}

impl BatchFirstDummy {
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

impl DataSource for BatchFirstDummy {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        "batch-first-dummy"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        self.distance_calls.fetch_add(1, Ordering::Relaxed);
        let a = self
            .data
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .data
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
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

        for ((left, right), slot) in pairs.iter().copied().zip(out.iter_mut()) {
            let a = self
                .data
                .get(left)
                .ok_or(DataSourceError::OutOfBounds { index: left })?;
            let b = self
                .data
                .get(right)
                .ok_or(DataSourceError::OutOfBounds { index: right })?;
            *slot = (a - b).abs();
        }
        Ok(())
    }
}

#[rstest]
fn batch_distances_prefers_distance_batch_override() {
    let batch_calls = Arc::new(AtomicUsize::new(0));
    let distance_calls = Arc::new(AtomicUsize::new(0));
    let source = BatchFirstDummy::new(
        vec![0.0, 2.0, 7.0],
        Arc::clone(&batch_calls),
        Arc::clone(&distance_calls),
    );

    let out = source
        .batch_distances(0, &[1, 2])
        .expect("batch distances must succeed");

    assert_eq!(out, vec![2.0, 7.0]);
    assert_eq!(
        batch_calls.load(Ordering::Relaxed),
        1,
        "batch override should be called",
    );
    assert_eq!(
        distance_calls.load(Ordering::Relaxed),
        0,
        "scalar distance should not be called via default batch_distances",
    );
}

#[rstest]
fn batch_distances_propagates_distance_batch_out_of_bounds() {
    let batch_calls = Arc::new(AtomicUsize::new(0));
    let distance_calls = Arc::new(AtomicUsize::new(0));
    let source = BatchFirstDummy::new(vec![0.0], batch_calls, distance_calls);

    let err = source
        .batch_distances(0, &[2])
        .expect_err("out-of-bounds must propagate");
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 2 }));
}
