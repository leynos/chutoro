//! Integration tests validating the distance helpers exported by `chutoro-core`.

use chutoro_core::{CosineNorms, DistanceError, VectorKind, cosine_distance, euclidean_distance};
use rstest::rstest;

#[rstest]
#[case(vec![0.0_f32, 0.0], vec![0.0_f32, 0.0], 0.0_f32)]
#[case(vec![1.0_f32, 2.0], vec![4.0_f32, 6.0], 5.0_f32)]
#[case(vec![1.0_f32, 2.0, 3.0], vec![4.0_f32, 6.0, 8.0], 50.0_f32.sqrt())]
fn euclidean_distance_returns_expected(
    #[case] left: Vec<f32>,
    #[case] right: Vec<f32>,
    #[case] expected: f32,
) {
    let distance = euclidean_distance(&left, &right).expect("distance should succeed");
    assert!((distance.value() - expected).abs() < 1e-6);
}

#[test]
fn euclidean_distance_rejects_dimension_mismatch() {
    let error =
        euclidean_distance(&[1.0_f32], &[1.0_f32, 2.0_f32]).expect_err("dimensions must match");
    assert!(matches!(
        error,
        DistanceError::DimensionMismatch { left: 1, right: 2 }
    ));
}

#[test]
fn euclidean_distance_rejects_zero_length() {
    let empty: [f32; 0] = [];
    let error = euclidean_distance(&empty, &empty).expect_err("empty input");
    assert!(matches!(error, DistanceError::ZeroLength));
}

#[test]
fn euclidean_distance_rejects_non_finite_values() {
    let error = euclidean_distance(&[f32::NAN], &[0.0_f32]).expect_err("reject NaN");
    match error {
        DistanceError::NonFinite {
            which: VectorKind::Left,
            index: 0,
            value,
        } => assert!(value.is_nan()),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[rstest]
#[case(vec![1.0_f32, 0.0], vec![1.0_f32, 0.0], 0.0_f32)]
#[case(vec![1.0_f32, 0.0], vec![-1.0_f32, 0.0], 2.0_f32)]
#[case(vec![1.0_f32, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0], 1.0_f32)]
fn cosine_distance_returns_expected(
    #[case] left: Vec<f32>,
    #[case] right: Vec<f32>,
    #[case] expected: f32,
) {
    let distance = cosine_distance(&left, &right, None).expect("distance should succeed");
    assert!((distance.value() - expected).abs() < 1e-6);
}

#[test]
fn cosine_distance_respects_precomputed_norms() {
    let left = vec![1.0_f32, 2.0, 3.0];
    let right = vec![4.0_f32, 5.0, 6.0];

    let baseline = cosine_distance(&left, &right, None).expect("baseline distance");
    let norms = CosineNorms::from_vectors(&left, &right).expect("norms from vectors");
    assert!((norms.left_norm().value() - norms.left()).abs() < f32::EPSILON);
    assert!((norms.right_norm().value() - norms.right()).abs() < f32::EPSILON);

    let cached = cosine_distance(&left, &right, Some(norms)).expect("cached distance");

    assert!((baseline.value() - cached.value()).abs() < 1e-6);
}

#[test]
fn cosine_distance_rejects_zero_magnitude_vectors() {
    let error = cosine_distance(&[0.0_f32, 0.0], &[1.0_f32, 0.0], None)
        .expect_err("zero magnitude must fail");
    assert!(matches!(
        error,
        DistanceError::ZeroMagnitude {
            which: VectorKind::Left
        }
    ));

    let norms = CosineNorms::new(1.0_f32, 1.0_f32).expect("valid norms");
    let error = cosine_distance(&[0.0_f32, 0.0], &[1.0_f32, 0.0], Some(norms))
        .expect_err("zero vector must fail even with cached norms");
    assert!(matches!(
        error,
        DistanceError::ZeroMagnitude {
            which: VectorKind::Left
        }
    ));
}

#[test]
fn cosine_distance_rejects_invalid_norms() {
    let error = CosineNorms::new(f32::NAN, 1.0_f32).expect_err("reject NaN norm");
    assert!(matches!(
        error,
        DistanceError::InvalidNorm {
            which: VectorKind::Left,
            value
        } if value.is_nan()
    ));

    let error = CosineNorms::new(0.0_f32, 1.0_f32).expect_err("reject zero norm");
    assert!(matches!(
        error,
        DistanceError::ZeroMagnitude {
            which: VectorKind::Left
        }
    ));

    let error = CosineNorms::new(-1.0_f32, 1.0_f32).expect_err("reject negative norm");
    assert!(matches!(
        error,
        DistanceError::InvalidNorm {
            which: VectorKind::Left,
            value: v
        } if v < 0.0
    ));
}

#[test]
fn cosine_distance_rejects_non_finite_values() {
    let error = cosine_distance(&[f32::INFINITY], &[1.0_f32], None).expect_err("reject infinity");
    assert!(matches!(
        error,
        DistanceError::NonFinite {
            which: VectorKind::Left,
            index: 0,
            value
        } if value.is_infinite()
    ));
}

// ============================================================================
// Distance Kernel Consistency Tests (Symmetry, Identity, Non-negativity)
// ============================================================================

#[rstest]
#[case::unit_vectors(&[1.0_f32, 0.0, 0.0], &[0.0_f32, 1.0, 0.0])]
#[case::identical_vectors(&[1.0_f32, 2.0, 3.0], &[1.0_f32, 2.0, 3.0])]
#[case::opposite_vectors(&[1.0_f32, 0.0], &[-1.0_f32, 0.0])]
#[case::arbitrary_vectors(&[3.5_f32, 2.7, 1.4], &[1.0_f32, 2.0, 3.0])]
fn euclidean_is_symmetric(#[case] a: &[f32], #[case] b: &[f32]) {
    let ab = euclidean_distance(a, b).expect("a to b should succeed");
    let ba = euclidean_distance(b, a).expect("b to a should succeed");
    assert!(
        (ab.value() - ba.value()).abs() < 1e-6,
        "euclidean distance should be symmetric: d(a,b)={} != d(b,a)={}",
        ab.value(),
        ba.value()
    );
}

#[rstest]
#[case::zeros(&[0.0_f32, 0.0, 0.0])]
#[case::unit(&[1.0_f32, 0.0, 0.0])]
#[case::arbitrary(&[3.5_f32, 2.7, 1.4])]
fn euclidean_zero_on_identical(#[case] v: &[f32]) {
    let d = euclidean_distance(v, v).expect("identical vectors should succeed");
    assert!(
        d.value().abs() < 1e-6,
        "euclidean distance to itself should be zero, got {}",
        d.value()
    );
}

#[rstest]
#[case::different_vectors(&[1.0_f32, 0.0], &[0.0_f32, 1.0])]
#[case::distant_vectors(&[0.0_f32, 0.0, 0.0], &[10.0_f32, 10.0, 10.0])]
fn euclidean_is_non_negative(#[case] a: &[f32], #[case] b: &[f32]) {
    let d = euclidean_distance(a, b).expect("distance should succeed");
    assert!(
        d.value() >= 0.0,
        "euclidean distance should be non-negative, got {}",
        d.value()
    );
}

#[rstest]
#[case::unit_vectors(&[1.0_f32, 0.0, 0.0], &[0.0_f32, 1.0, 0.0])]
#[case::identical_vectors(&[1.0_f32, 2.0, 3.0], &[1.0_f32, 2.0, 3.0])]
#[case::opposite_vectors(&[1.0_f32, 0.0], &[-1.0_f32, 0.0])]
#[case::arbitrary_vectors(&[3.5_f32, 2.7, 1.4], &[1.0_f32, 2.0, 3.0])]
fn cosine_is_symmetric(#[case] a: &[f32], #[case] b: &[f32]) {
    let ab = cosine_distance(a, b, None).expect("a to b should succeed");
    let ba = cosine_distance(b, a, None).expect("b to a should succeed");
    assert!(
        (ab.value() - ba.value()).abs() < 1e-6,
        "cosine distance should be symmetric: d(a,b)={} != d(b,a)={}",
        ab.value(),
        ba.value()
    );
}

#[rstest]
#[case::unit(&[1.0_f32, 0.0, 0.0])]
#[case::arbitrary(&[3.5_f32, 2.7, 1.4])]
#[case::negative(&[-1.0_f32, -2.0, -3.0])]
fn cosine_zero_on_identical(#[case] v: &[f32]) {
    let d = cosine_distance(v, v, None).expect("identical vectors should succeed");
    assert!(
        d.value().abs() < 1e-6,
        "cosine distance to itself should be zero, got {}",
        d.value()
    );
}

#[rstest]
#[case::orthogonal(&[1.0_f32, 0.0], &[0.0_f32, 1.0])]
#[case::parallel(&[1.0_f32, 2.0, 3.0], &[2.0_f32, 4.0, 6.0])]
#[case::opposite(&[1.0_f32, 0.0], &[-1.0_f32, 0.0])]
fn cosine_is_bounded_zero_to_two(#[case] a: &[f32], #[case] b: &[f32]) {
    let d = cosine_distance(a, b, None).expect("distance should succeed");
    assert!(
        d.value() >= 0.0 && d.value() <= 2.0,
        "cosine distance should be in [0, 2], got {}",
        d.value()
    );
}
