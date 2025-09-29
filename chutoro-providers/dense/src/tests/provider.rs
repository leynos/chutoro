use super::{DenseMatrixProvider, DenseMatrixProviderError, support::*};
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
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

#[rstest]
fn matrix_provider_rejects_null_rows() {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
    builder.values().append_value(1.0);
    builder.values().append_value(2.0);
    builder.append(true);
    builder.values().append_null();
    builder.values().append_null();
    builder.append(false);
    let array = builder.finish();
    let err = DenseMatrixProvider::try_from_fixed_size_list("demo", &array)
        .expect_err("null rows must be rejected");
    assert!(matches!(err, DenseMatrixProviderError::NullRow { row: 1 }));
}

#[rstest]
fn matrix_provider_rejects_null_values() {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 2);
    builder.values().append_value(1.0);
    builder.values().append_value(2.0);
    builder.append(true);
    builder.values().append_value(3.0);
    builder.values().append_null();
    builder.append(true);
    let array = builder.finish();
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
