//! Order-sensitive `DataSource` fixture for the default batch helpers.
//!
//! The other fixtures in this crate compute an absolute difference, so
//! `distance(a, b)` and `distance(b, a)` are indistinguishable and a pair that
//! is forwarded backwards still produces the expected value. These tests use a
//! source whose result changes when a pair is reversed, which is what lets them
//! prove that `distance_batch` and `batch_distances` forward `left` and `right`
//! in the order supplied rather than merely forwarding some pair.

use super::super::{DataSource, DataSourceError, PointPair};
use rstest::{fixture, rstest};

/// Weight applied to the left endpoint so the result depends on pair order.
const LEFT_WEIGHT: f32 = 4.0;

/// `DataSource` whose distance depends on which endpoint is `left`.
#[derive(Clone, Debug)]
struct OrderedSource {
    data: Vec<f32>,
}

impl OrderedSource {
    /// Builds a fixture over the supplied scalar values.
    fn new(data: Vec<f32>) -> Self {
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

impl DataSource for OrderedSource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &'static str {
        "ordered"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_value = self.value(left)?;
        let right_value = self.value(right)?;
        Ok(left_value.mul_add(LEFT_WEIGHT, right_value))
    }
}

/// Builds the shared fixture with values that make reversal observable.
#[fixture]
fn ordered_source() -> OrderedSource {
    OrderedSource::new(vec![1.0, 3.0, 6.0])
}

/// The default `distance_batch` must forward each pair's endpoints in order.
///
/// Reversing a pair must change the result, so a reversed forwarding would
/// fail the assertion instead of silently producing the same value.
#[rstest]
fn distance_batch_forwards_left_and_right_in_order(ordered_source: OrderedSource) {
    let pairs = [PointPair::new(0, 1), PointPair::new(1, 0)];
    let mut out = [0.0_f32; 2];

    ordered_source
        .distance_batch(&pairs, &mut out)
        .expect("default distance_batch must succeed");

    let forward = ordered_source
        .distance(0, 1)
        .expect("forward distance must succeed");
    let reverse = ordered_source
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

/// The default `batch_distances` must pass the query as the left endpoint.
#[rstest]
fn batch_distances_passes_query_as_left_endpoint(ordered_source: OrderedSource) {
    let distances = ordered_source
        .batch_distances(0, &[1, 2])
        .expect("default batch_distances must succeed");

    let expected: Vec<u32> = [1, 2]
        .into_iter()
        .map(|candidate| {
            ordered_source
                .distance(0, candidate)
                .expect("forward distance must succeed")
                .to_bits()
        })
        .collect();
    let reversed: Vec<u32> = [1, 2]
        .into_iter()
        .map(|candidate| {
            ordered_source
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
        expected
    );
    assert_ne!(
        expected, reversed,
        "the fixture must distinguish the query from the candidate",
    );
}
