use chutoro_core::{DataSource, DataSourceError};
use rstest::rstest;

struct Dummy(Vec<f32>);

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

#[rstest]
fn distance_batch_returns_distances() {
    let src = Dummy(vec![1.0, 3.0, 6.0]);
    let pairs = [(0, 1), (1, 2)];
    let mut out = [0.0; 2];
    src.distance_batch(&pairs, &mut out).unwrap();
    assert_eq!(out, [2.0, 3.0]);
}

#[rstest]
#[case::short_output(vec![(0, 1)], vec![])]
#[case::long_output(vec![(0, 1)], vec![0.0, 0.0])]
fn distance_batch_length_mismatch(#[case] pairs: Vec<(usize, usize)>, #[case] mut out: Vec<f32>) {
    let src = Dummy(vec![1.0, 2.0]);
    let err = src.distance_batch(&pairs, &mut out).unwrap_err();
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
}

#[rstest]
fn distance_out_of_bounds() {
    let src = Dummy(vec![1.0, 2.0]);
    let err = src.distance(0, 2).unwrap_err();
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 2 }));
}
