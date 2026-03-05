//! Tests for SIMD-aware Euclidean distance kernels.

use super::kernels;
use super::*;
use rstest::{fixture, rstest};

fn close(left: Distance, right: Distance) {
    let left = left.get();
    let right = right.get();
    let tolerance = 1.0e-6_f32;
    assert!(
        (left - right).abs() <= tolerance,
        "left={left}, right={right}, tolerance={tolerance}",
    );
}

#[fixture]
fn matrix_3x2() -> RowMajorMatrix<'static> {
    const VALUES: [f32; 6] = [1.0, 2.0, 4.0, 6.0, 2.0, 1.0];
    RowMajorMatrix::new(
        MatrixValues::new(&VALUES),
        RowCount::new(3),
        Dimension::new(2),
    )
}

#[rstest]
#[case(vec![0.0], vec![1.0])]
#[case(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 8.0])]
#[case(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5.0, 4.0, 3.0, 2.0, 1.0])]
#[case(
    vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    vec![2.0, 3.0, 5.0, 9.0, 17.0, 33.0, 65.0, 129.0, 257.0],
)]
fn euclidean_distance_matches_scalar_reference(#[case] left: Vec<f32>, #[case] right: Vec<f32>) {
    let expected = kernels::euclidean_distance_scalar(&left, &right);
    let actual = euclidean_distance(RowSlice::new(&left), RowSlice::new(&right));
    close(actual, Distance::new(expected));
}

#[rstest]
#[case(
    vec![DistancePair::new(RowIndex::new(0), RowIndex::new(1))],
    vec![]
)]
#[case(
    vec![DistancePair::new(RowIndex::new(0), RowIndex::new(1))],
    vec![0.0, 1.0]
)]
fn batch_pairs_reject_mismatched_output_lengths(
    #[case] pairs: Vec<DistancePair>,
    #[case] mut out: Vec<f32>,
) {
    let matrix = RowMajorMatrix::new(
        MatrixValues::new(&[1.0, 2.0, 3.0, 4.0]),
        RowCount::new(2),
        Dimension::new(2),
    );
    let mut out_buffer = DistanceBuffer::new(&mut out);
    let err = euclidean_distance_batch_pairs(matrix, &pairs, &mut out_buffer)
        .expect_err("mismatched outputs must fail");
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
}

#[rstest]
fn batch_pairs_compute_distances(matrix_3x2: RowMajorMatrix<'static>) {
    let pairs = vec![
        DistancePair::new(RowIndex::new(0), RowIndex::new(1)),
        DistancePair::new(RowIndex::new(0), RowIndex::new(2)),
        DistancePair::new(RowIndex::new(2), RowIndex::new(1)),
    ];
    let mut out = vec![0.0_f32; pairs.len()];
    let mut out_buffer = DistanceBuffer::new(&mut out);

    euclidean_distance_batch_pairs(matrix_3x2, &pairs, &mut out_buffer)
        .expect("batch computation must succeed");

    close(Distance::new(out[0]), Distance::new(5.0_f32));
    close(Distance::new(out[1]), Distance::new((2.0_f32).sqrt()));
    close(Distance::new(out[2]), Distance::new((29.0_f32).sqrt()));
}

#[rstest]
fn batch_pairs_leave_output_unmodified_on_error(matrix_3x2: RowMajorMatrix<'static>) {
    let pairs = vec![
        DistancePair::new(RowIndex::new(0), RowIndex::new(1)),
        DistancePair::new(RowIndex::new(0), RowIndex::new(9)),
    ];
    let mut out = vec![10.0_f32, 20.0_f32];
    let mut out_buffer = DistanceBuffer::new(&mut out);

    let err = euclidean_distance_batch_pairs(matrix_3x2, &pairs, &mut out_buffer)
        .expect_err("out-of-bounds pair must fail");

    assert!(matches!(err, DataSourceError::OutOfBounds { index: 9 }));
    assert_eq!(out, vec![10.0_f32, 20.0_f32]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx2_kernel_matches_scalar_when_available() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    let left: Vec<f32> = (0_u32..35_u32).map(|index| index as f32 * 0.5).collect();
    let right: Vec<f32> = (0_u32..35_u32)
        .map(|index| (35_u32 - index) as f32 * 0.25)
        .collect();

    let expected = kernels::euclidean_distance_scalar(&left, &right);
    let actual = kernels::euclidean_distance_avx2_entry(&left, &right);
    close(Distance::new(actual), Distance::new(expected));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn avx512_entrypoint_matches_scalar_when_available() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        return;
    }

    let left: Vec<f32> = (0_u32..67_u32).map(|index| index as f32 * 0.125).collect();
    let right: Vec<f32> = (0_u32..67_u32)
        .map(|index| (67_u32 - index) as f32 * 0.375)
        .collect();

    let expected = kernels::euclidean_distance_scalar(&left, &right);
    let actual = kernels::euclidean_distance_avx512_entry(&left, &right);
    close(Distance::new(actual), Distance::new(expected));
}
