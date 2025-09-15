#![expect(clippy::expect_used, reason = "tests require contextual panics")]
//! Integration tests for the DataSource trait behaviour.
use chutoro_core::{DataSource, DataSourceError};
use rstest::{fixture, rstest};

struct Dummy(Vec<f32>);

#[fixture]
fn dummy(#[default(vec![1.0, 2.0])] data: Vec<f32>) -> Dummy {
    Dummy(data)
}

impl DataSource for Dummy {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn name(&self) -> &str {
        "dummy"
    }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self
            .0
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .0
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
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
