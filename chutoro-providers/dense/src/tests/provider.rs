use super::{DenseMatrixProvider, DenseMatrixProviderError, support::*};
use arrow_array::{ArrayRef, FixedSizeListArray};
use arrow_schema::{DataType, Field};
use chutoro_core::DataSource;
use rstest::rstest;
use std::sync::Arc;

#[rstest]
fn matrix_provider_from_fixed_size_list() {
    let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let provider =
        DenseMatrixProvider::try_from_fixed_size_list("demo", &array).expect("valid matrix");
    assert_eq!(provider.len(), 2);
    assert_eq!(provider.dimension(), 3);
    assert_eq!(provider.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let distance = provider.distance(0, 1).expect("distance should work");
    assert!((distance - (27.0_f32).sqrt()).abs() < 1e-6);
}

#[rstest]
fn matrix_provider_distance_batch() {
    let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let provider =
        DenseMatrixProvider::try_from_fixed_size_list("demo", &array).expect("valid matrix");
    let pairs = vec![(0, 1), (1, 0)];
    let mut out = vec![0.0; pairs.len()];
    provider
        .distance_batch(&pairs, &mut out)
        .expect("batch distances should work");
    for value in out {
        assert!((value - (27.0_f32).sqrt()).abs() < 1e-6);
    }
}

#[rstest]
#[case::odd_dimension(
    vec![
        vec![1.0, 3.0, 5.0, 7.0, 9.0],
        vec![2.0, 4.0, 6.0, 8.0, 10.0],
        vec![0.5, 1.5, 2.5, 3.5, 4.5],
    ],
    vec![(0, 1), (0, 2), (2, 1)],
)]
#[case::avx2_tail(
    vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0],
        vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0],
    ],
    vec![(0, 1), (1, 2), (2, 0), (1, 0)],
)]
#[case::avx512_tail(
    vec![
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0,
        ],
        vec![
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            1.0, 0.0,
        ],
        vec![
            1.0, 1.5, 2.5, 4.0, 6.5, 10.5, 17.0, 27.5, 44.5, 72.0, 116.5, 188.5, 305.0, 493.5,
            798.5, 1292.0, 2090.5,
        ],
    ],
    vec![(0, 1), (0, 2), (2, 1)],
)]
fn matrix_provider_distance_batch_matches_scalar_reference(
    #[case] rows: Vec<Vec<f32>>,
    #[case] pairs: Vec<(usize, usize)>,
) {
    let dimension = rows
        .first()
        .map(Vec::len)
        .expect("rows must include at least one vector");
    let flat_values: Vec<f32> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    let provider = DenseMatrixProvider::from_parts("simd-demo", rows.len(), dimension, flat_values);
    let mut out = vec![0.0_f32; pairs.len()];
    provider
        .distance_batch(&pairs, &mut out)
        .expect("batch distances should succeed");

    let expected: Vec<f32> = pairs
        .iter()
        .map(|(left, right)| scalar_distance(&rows[*left], &rows[*right]))
        .collect();
    assert_eq!(out.len(), expected.len());
    for (actual, expected_value) in out.iter().copied().zip(expected.into_iter()) {
        assert!(
            (actual - expected_value).abs() <= 1.0e-6_f32,
            "actual={actual}, expected={expected_value}",
        );
    }
}

#[rstest]
fn matrix_provider_distance_out_of_bounds() {
    let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let provider =
        DenseMatrixProvider::try_from_fixed_size_list("demo", &array).expect("valid matrix");
    let err = provider
        .distance(0, 99)
        .expect_err("distance must report out-of-bounds");
    assert!(matches!(
        err,
        chutoro_core::DataSourceError::OutOfBounds { index: 99 }
    ));
}

#[rstest]
fn matrix_provider_distance_batch_length_mismatch() {
    let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let provider =
        DenseMatrixProvider::try_from_fixed_size_list("demo", &array).expect("valid matrix");
    let pairs = vec![(0, 1)];
    let mut out = vec![0.0; 2];
    let err = provider
        .distance_batch(&pairs, &mut out)
        .expect_err("mismatched lengths must error");
    assert!(matches!(
        err,
        chutoro_core::DataSourceError::OutputLengthMismatch {
            out: 2,
            expected: 1
        }
    ));
}

#[rstest]
fn matrix_provider_distance_batch_empty() {
    let array = build_array(&[[1.0, 2.0, 3.0]]);
    let provider =
        DenseMatrixProvider::try_from_fixed_size_list("demo", &array).expect("valid matrix");
    let pairs: Vec<(usize, usize)> = Vec::new();
    let mut out = Vec::new();
    provider
        .distance_batch(&pairs, &mut out)
        .expect("empty batch must succeed");
    assert!(out.is_empty());
}

fn scalar_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| {
            let delta = l - r;
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
}

#[rstest]
fn matrix_provider_rejects_null_rows() {
    let rows = vec![Some(vec![1.0, 2.0]), None];
    let array = build_list_array_with_row_nulls(&rows, 2);
    let err = DenseMatrixProvider::try_from_fixed_size_list("demo", &array)
        .expect_err("null rows must be rejected");
    assert!(matches!(err, DenseMatrixProviderError::NullRow { row: 1 }));
}

#[rstest]
fn matrix_provider_rejects_null_values() {
    let rows = vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), None]];
    let array = build_list_array_with_value_nulls(&rows, 2);
    let err = DenseMatrixProvider::try_from_fixed_size_list("demo", &array)
        .expect_err("null values must be rejected");
    assert!(matches!(
        err,
        DenseMatrixProviderError::NullValue {
            row: 1,
            value_index: 1
        }
    ));
}

#[rstest]
fn matrix_provider_rejects_non_float_children() {
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let values: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3, 4]));
    let array = FixedSizeListArray::new(field, 2, values, None);
    let err = DenseMatrixProvider::try_from_fixed_size_list("demo", &array)
        .expect_err("non-float children must be rejected");
    assert!(matches!(
        err,
        DenseMatrixProviderError::InvalidListValueType { .. }
    ));
}
