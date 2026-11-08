//! Integration tests for the `DataSource` trait behaviour.

mod common;

use chutoro_core::{DataSource, DataSourceError, PointPair};
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

/// Weight applied to the left endpoint so [`OrderedDummy`] is order-sensitive.
const LEFT_WEIGHT: f32 = 4.0;

/// `DataSource` fixture whose distance depends on endpoint order.
///
/// [`Dummy`] computes an absolute difference, so it cannot tell `left` from
/// `right`. This fixture weights the left endpoint more heavily, which lets an
/// assertion fail if a batch helper forwards the endpoints of a pair backwards.
#[derive(Clone)]
struct OrderedDummy {
    data: Vec<f32>,
}

impl OrderedDummy {
    /// Builds a fixture over the supplied scalar values.
    const fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Looks up one endpoint, reporting the index that was out of bounds.
    fn value(&self, index: usize) -> Result<f32, DataSourceError> {
        self.data
            .get(index)
            .copied()
            .ok_or(DataSourceError::OutOfBounds { index })
    }
}

impl DataSource for OrderedDummy {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &'static str {
        "ordered-dummy"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_value = self.value(left)?;
        let right_value = self.value(right)?;
        Ok(left_value.mul_add(LEFT_WEIGHT, right_value))
    }
}

#[rstest(dummy(vec![1.0, 3.0, 6.0]))]
fn distance_batch_returns_distances(dummy: Dummy) {
    let pairs = [PointPair::new(0, 1), PointPair::new(1, 2)];
    let mut out = [0.0; 2];
    dummy
        .distance_batch(&pairs, &mut out)
        .expect("distance_batch must succeed");
    assert_eq!(
        out.map(f32::to_bits),
        [2.0_f32.to_bits(), 3.0_f32.to_bits()]
    );
}

/// The public `distance_batch` contract must not reorder a pair's endpoints.
///
/// [`OrderedDummy`] returns a different value for a reversed pair, so this test
/// fails if the default implementation (or an override) swaps `left` and
/// `right`.
#[rstest]
fn distance_batch_forwards_left_and_right_in_order() {
    let source = OrderedDummy::new(vec![1.0, 3.0, 6.0]);
    let pairs = [PointPair::new(0, 1), PointPair::new(1, 0)];
    let mut out = [0.0_f32; 2];

    source
        .distance_batch(&pairs, &mut out)
        .expect("distance_batch must succeed");

    let forward = source
        .distance(0, 1)
        .expect("forward distance must succeed");
    let reverse = source
        .distance(1, 0)
        .expect("reverse distance must succeed");

    assert_eq!(
        out.map(f32::to_bits),
        [forward.to_bits(), reverse.to_bits()]
    );
    assert_ne!(
        forward.to_bits(),
        reverse.to_bits(),
        "the fixture must produce different results for reversed endpoints",
    );
}

/// `batch_distances` must pass the query as the left endpoint of every pair.
#[rstest]
fn batch_distances_passes_query_as_left_endpoint() {
    let source = OrderedDummy::new(vec![1.0, 3.0, 6.0]);

    let distances = source
        .batch_distances(0, &[1, 2])
        .expect("batch_distances must succeed");

    let forwards: Vec<u32> = [1, 2]
        .into_iter()
        .map(|candidate| {
            source
                .distance(0, candidate)
                .expect("forward distance must succeed")
                .to_bits()
        })
        .collect();
    let reversals: Vec<u32> = [1, 2]
        .into_iter()
        .map(|candidate| {
            source
                .distance(candidate, 0)
                .expect("reverse distance must succeed")
                .to_bits()
        })
        .collect();

    assert_eq!(
        distances
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        forwards,
    );
    assert_ne!(
        forwards, reversals,
        "the fixture must distinguish the query from the candidate",
    );
}

#[rstest]
#[case::short_output(vec![(0, 1)], vec![])]
#[case::long_output(vec![(0, 1)], vec![0.0, 0.0])]
fn distance_batch_length_mismatch(
    dummy: Dummy,
    #[case] pairs: Vec<(usize, usize)>,
    #[case] mut out: Vec<f32>,
) {
    let point_pairs: Vec<PointPair> = pairs.into_iter().map(PointPair::from).collect();
    let err = dummy
        .distance_batch(&point_pairs, &mut out)
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
    let pairs = [PointPair::new(0, 1), PointPair::new(0, 99)];
    let mut out = [1.0_f32, 1.0];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .expect_err("distance_batch must propagate inner errors");
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 99 }));
    assert_eq!(
        out.map(f32::to_bits),
        [1.0_f32.to_bits(), 1.0_f32.to_bits()]
    );
}

#[rstest(dummy(vec![]))]
fn distance_batch_empty_ok(dummy: Dummy) {
    let pairs: [PointPair; 0] = [];
    let mut out: [f32; 0] = [];
    assert!(dummy.distance_batch(&pairs, &mut out).is_ok());
}

#[rstest(dummy(vec![1.0]))]
fn distance_batch_empty_pairs_error(dummy: Dummy) {
    let pairs: [PointPair; 0] = [];
    let mut out = [0.0];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .expect_err("distance_batch must report length mismatch");
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
}

#[rstest]
fn distance_batch_empty_output_error(dummy: Dummy) {
    let pairs = [PointPair::new(0, 1)];
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
    const fn new(
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

    fn name(&self) -> &'static str {
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
        Ok(a.mul_add(1.0, std::ops::Neg::neg(*b)).abs())
    }

    fn distance_batch(&self, pairs: &[PointPair], out: &mut [f32]) -> Result<(), DataSourceError> {
        self.batch_calls.fetch_add(1, Ordering::Relaxed);
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }

        for (pair, slot) in pairs.iter().zip(out.iter_mut()) {
            let left = pair.left();
            let right = pair.right();
            let a = self
                .data
                .get(left)
                .ok_or(DataSourceError::OutOfBounds { index: left })?;
            let b = self
                .data
                .get(right)
                .ok_or(DataSourceError::OutOfBounds { index: right })?;
            *slot = a.mul_add(1.0, std::ops::Neg::neg(*b)).abs();
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
