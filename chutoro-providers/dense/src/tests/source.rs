use super::DenseSource;
use chutoro_core::{DataSource, DataSourceError};
use rstest::rstest;

#[rstest]
fn distance_dimension_mismatch() {
    let ds = DenseSource::from_parts("d", vec![vec![0.0], vec![1.0, 2.0]]);
    let err = ds
        .distance(0, 1)
        .expect_err("distance must validate dimensions");
    assert!(matches!(err, DataSourceError::DimensionMismatch { .. }));
}

#[rstest]
fn try_new_rejects_mismatched_rows() {
    let err = DenseSource::try_new("d", vec![vec![0.0], vec![1.0, 2.0]]);
    assert!(matches!(
        err,
        Err(DataSourceError::DimensionMismatch { .. })
    ));
}

#[rstest]
fn try_new_rejects_empty_data() {
    let err = DenseSource::try_new("d", Vec::new());
    assert!(matches!(err, Err(DataSourceError::EmptyData)));
}

#[rstest]
fn try_new_rejects_zero_dimension() {
    let err = DenseSource::try_new("d", vec![Vec::new()]);
    assert!(matches!(err, Err(DataSourceError::ZeroDimension)));
}

#[rstest]
fn distance_out_of_bounds() {
    let ds = DenseSource::try_new("d", vec![vec![0.0], vec![1.0]]).expect("rows must match");
    let err = ds
        .distance(0, 99)
        .expect_err("distance must report out-of-bounds");
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 99 }));
}

#[rstest]
fn distance_ok() {
    let ds = DenseSource::try_new("d", vec![vec![0.0, 0.0], vec![3.0, 4.0]])
        .expect("valid uniform rows");
    let d = ds.distance(0, 1).expect("distance must succeed");
    assert!((d - 5.0).abs() < 1e-6);
}

#[rstest]
fn distance_batch_empty_pairs() {
    let ds = DenseSource::try_new("d", vec![vec![0.0, 0.0]]).expect("single row should be valid");
    let pairs: Vec<(usize, usize)> = Vec::new();

    let mut out = Vec::new();
    ds.distance_batch(&pairs, &mut out)
        .expect("empty batches must be allowed");
    assert!(out.is_empty());

    let mut prefilled = vec![42.0, 99.0];
    prefilled.clear();
    ds.distance_batch(&pairs, &mut prefilled)
        .expect("empty batches must leave cleared buffers untouched");
    assert!(prefilled.is_empty());
}

#[rstest]
fn distance_batch_output_length_mismatch() {
    let ds = DenseSource::try_new("d", vec![vec![0.0, 0.0], vec![1.0, 1.0]])
        .expect("valid uniform rows");
    let pairs = vec![(0, 1)];
    let mut out = vec![0.0, 0.0];
    let err = ds
        .distance_batch(&pairs, &mut out)
        .expect_err("mismatched output lengths must error");
    assert!(matches!(
        err,
        DataSourceError::OutputLengthMismatch {
            out: 2,
            expected: 1
        }
    ));
    assert_eq!(out, vec![0.0, 0.0]);
}
