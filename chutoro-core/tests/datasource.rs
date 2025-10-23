//! Integration tests for the DataSource trait behaviour.

mod common;

use anyhow::{Context, Result};
use chutoro_core::{DataSource, DataSourceError};
use common::Dummy;
use rstest::{fixture, rstest};

type TestResult<T = ()> = Result<T>;

#[fixture]
fn dummy(#[default(vec![1.0, 2.0])] data: Vec<f32>) -> Dummy {
    Dummy::new(data)
}

#[rstest(dummy(vec![1.0, 3.0, 6.0]))]
fn distance_batch_returns_distances(dummy: Dummy) -> TestResult {
    let pairs = [(0, 1), (1, 2)];
    let mut out = [0.0; 2];
    dummy
        .distance_batch(&pairs, &mut out)
        .context("distance_batch must succeed")?;
    assert_eq!(out, [2.0, 3.0]);
    Ok(())
}

#[rstest]
#[case::short_output(vec![(0, 1)], vec![])]
#[case::long_output(vec![(0, 1)], vec![0.0, 0.0])]
fn distance_batch_length_mismatch(
    dummy: Dummy,
    #[case] pairs: Vec<(usize, usize)>,
    #[case] mut out: Vec<f32>,
) -> TestResult {
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .err()
        .context("distance_batch must report length mismatch")?;
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
    Ok(())
}

#[rstest]
fn distance_out_of_bounds(dummy: Dummy) -> TestResult {
    let err = dummy
        .distance(0, 2)
        .err()
        .context("distance must check bounds")?;
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 2 }));
    Ok(())
}

#[rstest]
fn distance_batch_preserves_out_on_error(dummy: Dummy) -> TestResult {
    let pairs = [(0, 1), (0, 99)];
    let mut out = [1.0_f32, 1.0];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .err()
        .context("distance_batch must propagate inner errors")?;
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 99 }));
    assert_eq!(out, [1.0, 1.0]);
    Ok(())
}

#[rstest(dummy(vec![]))]
fn distance_batch_empty_ok(dummy: Dummy) -> TestResult {
    let pairs: [(usize, usize); 0] = [];
    let mut out: [f32; 0] = [];
    dummy
        .distance_batch(&pairs, &mut out)
        .context("distance_batch must allow empty input")?;
    Ok(())
}

#[rstest(dummy(vec![1.0]))]
fn distance_batch_empty_pairs_error(dummy: Dummy) -> TestResult {
    let pairs: [(usize, usize); 0] = [];
    let mut out = [0.0];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .err()
        .context("distance_batch must report length mismatch when pairs and output diverge")?;
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
    Ok(())
}

#[rstest]
fn distance_batch_empty_output_error(dummy: Dummy) -> TestResult {
    let pairs = [(0, 1)];
    let mut out: [f32; 0] = [];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .err()
        .context("distance_batch must report length mismatch for empty output buffer")?;
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
    Ok(())
}

#[rstest(dummy(vec![1.0, 2.0, 3.0]))]
fn distance_batch_index_bounds(dummy: Dummy) -> TestResult {
    let pairs = [(0, 3)];
    let mut out = [0.0];
    let err = dummy
        .distance_batch(&pairs, &mut out)
        .err()
        .context("distance_batch must check bounds for indices")?;
    assert!(matches!(err, DataSourceError::OutOfBounds { index: 3 }));
    Ok(())
}
