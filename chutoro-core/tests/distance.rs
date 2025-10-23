//! Integration tests validating the distance helpers exported by `chutoro-core`.

use anyhow::{Context, Result};
use chutoro_core::{CosineNorms, DistanceError, VectorKind, cosine_distance, euclidean_distance};
use rstest::rstest;

type TestResult<T = ()> = Result<T>;

#[rstest]
#[case(vec![0.0_f32, 0.0], vec![0.0_f32, 0.0], 0.0_f32)]
#[case(vec![1.0_f32, 2.0], vec![4.0_f32, 6.0], 5.0_f32)]
#[case(vec![1.0_f32, 2.0, 3.0], vec![4.0_f32, 6.0, 8.0], 50.0_f32.sqrt())]
fn euclidean_distance_returns_expected(
    #[case] left: Vec<f32>,
    #[case] right: Vec<f32>,
    #[case] expected: f32,
) -> TestResult {
    let distance = euclidean_distance(&left, &right).context("distance should succeed")?;
    assert!((distance.value() - expected).abs() < 1e-6);
    Ok(())
}

#[test]
fn euclidean_distance_rejects_dimension_mismatch() -> TestResult {
    let error = euclidean_distance(&[1.0_f32], &[1.0_f32, 2.0_f32])
        .err()
        .context("dimensions must match")?;
    assert!(matches!(
        error,
        DistanceError::DimensionMismatch { left: 1, right: 2 }
    ));
    Ok(())
}

#[test]
fn euclidean_distance_rejects_zero_length() -> TestResult {
    let empty: [f32; 0] = [];
    let error = euclidean_distance(&empty, &empty)
        .err()
        .context("empty input must fail")?;
    assert!(matches!(error, DistanceError::ZeroLength));
    Ok(())
}

#[test]
fn euclidean_distance_rejects_non_finite_values() -> TestResult {
    let error = euclidean_distance(&[f32::NAN], &[0.0_f32])
        .err()
        .context("reject NaN")?;
    match error {
        DistanceError::NonFinite {
            which: VectorKind::Left,
            index: 0,
            value,
        } => assert!(value.is_nan()),
        other => panic!("unexpected error: {other:?}"),
    }
    Ok(())
}

#[rstest]
#[case(vec![1.0_f32, 0.0], vec![1.0_f32, 0.0], 0.0_f32)]
#[case(vec![1.0_f32, 0.0], vec![-1.0_f32, 0.0], 2.0_f32)]
#[case(vec![1.0_f32, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0], 1.0_f32)]
fn cosine_distance_returns_expected(
    #[case] left: Vec<f32>,
    #[case] right: Vec<f32>,
    #[case] expected: f32,
) -> TestResult {
    let distance = cosine_distance(&left, &right, None).context("distance should succeed")?;
    assert!((distance.value() - expected).abs() < 1e-6);
    Ok(())
}

#[test]
fn cosine_distance_respects_precomputed_norms() -> TestResult {
    let left = vec![1.0_f32, 2.0, 3.0];
    let right = vec![4.0_f32, 5.0, 6.0];

    let baseline = cosine_distance(&left, &right, None).context("baseline distance")?;
    let norms = CosineNorms::from_vectors(&left, &right).context("norms from vectors")?;
    assert!((norms.left_norm().value() - norms.left()).abs() < f32::EPSILON);
    assert!((norms.right_norm().value() - norms.right()).abs() < f32::EPSILON);

    let cached = cosine_distance(&left, &right, Some(norms)).context("cached distance")?;

    assert!((baseline.value() - cached.value()).abs() < 1e-6);
    Ok(())
}

#[test]
fn cosine_distance_rejects_zero_magnitude_vectors() -> TestResult {
    let error = cosine_distance(&[0.0_f32, 0.0], &[1.0_f32, 0.0], None)
        .err()
        .context("zero magnitude must fail")?;
    assert!(matches!(
        error,
        DistanceError::ZeroMagnitude {
            which: VectorKind::Left
        }
    ));

    let norms = CosineNorms::new(1.0_f32, 1.0_f32).context("valid norms")?;
    let error = cosine_distance(&[0.0_f32, 0.0], &[1.0_f32, 0.0], Some(norms))
        .err()
        .context("zero vector must fail even with cached norms")?;
    assert!(matches!(
        error,
        DistanceError::ZeroMagnitude {
            which: VectorKind::Left
        }
    ));
    Ok(())
}

#[test]
fn cosine_distance_rejects_invalid_norms() -> TestResult {
    let error = CosineNorms::new(f32::NAN, 1.0_f32)
        .err()
        .context("reject NaN norm")?;
    assert!(matches!(
        error,
        DistanceError::InvalidNorm {
            which: VectorKind::Left,
            value
        } if value.is_nan()
    ));

    let error = CosineNorms::new(0.0_f32, 1.0_f32)
        .err()
        .context("reject zero norm")?;
    assert!(matches!(
        error,
        DistanceError::ZeroMagnitude {
            which: VectorKind::Left
        }
    ));

    let error = CosineNorms::new(-1.0_f32, 1.0_f32)
        .err()
        .context("reject negative norm")?;
    assert!(matches!(
        error,
        DistanceError::InvalidNorm {
            which: VectorKind::Left,
            value: v
        } if v < 0.0
    ));
    Ok(())
}

#[test]
fn cosine_distance_rejects_non_finite_values() -> TestResult {
    let error = cosine_distance(&[f32::INFINITY], &[1.0_f32], None)
        .err()
        .context("reject infinity")?;
    assert!(matches!(
        error,
        DistanceError::NonFinite {
            which: VectorKind::Left,
            index: 0,
            value
        } if value.is_infinite()
    ));
    Ok(())
}
