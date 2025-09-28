use super::{DenseMatrixProvider, DenseMatrixProviderError, DenseSource};
use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use chutoro_core::DataSource;
use parquet::arrow::arrow_writer::ArrowWriter;
use rstest::rstest;
use std::sync::Arc;

#[rstest]
fn distance_dimension_mismatch() {
    let ds = DenseSource::from_parts("d", vec![vec![0.0], vec![1.0, 2.0]]);
    let err = ds
        .distance(0, 1)
        .expect_err("distance must validate dimensions");
    assert!(matches!(
        err,
        chutoro_core::DataSourceError::DimensionMismatch { .. }
    ));
}

#[rstest]
fn try_new_rejects_mismatched_rows() {
    let err = DenseSource::try_new("d", vec![vec![0.0], vec![1.0, 2.0]]);
    assert!(matches!(
        err,
        Err(chutoro_core::DataSourceError::DimensionMismatch { .. })
    ));
}

#[rstest]
fn distance_out_of_bounds() {
    let ds = DenseSource::try_new("d", vec![vec![0.0], vec![1.0]]).expect("rows must match");
    let err = ds
        .distance(0, 99)
        .expect_err("distance must report out-of-bounds");
    assert!(matches!(
        err,
        chutoro_core::DataSourceError::OutOfBounds { index: 99 }
    ));
}

#[rstest]
fn distance_ok() {
    let ds = DenseSource::try_new("d", vec![vec![0.0, 0.0], vec![3.0, 4.0]])
        .expect("valid uniform rows");
    let d = ds.distance(0, 1).expect("distance must succeed");
    assert!((d - 5.0).abs() < 1e-6);
}

fn build_array(rows: &[[f32; 3]]) -> FixedSizeListArray {
    let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 3);
    for row in rows {
        for value in row {
            builder.values().append_value(*value);
        }
        builder.append(true);
    }
    builder.finish()
}

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

fn write_parquet(array: FixedSizeListArray) -> Bytes {
    let field = Field::new(
        "features",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3),
        false,
    );
    let schema = Arc::new(Schema::new(vec![field]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(array) as ArrayRef]).expect("batch");
    let mut buffer = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
    }
    Bytes::from(buffer)
}

#[rstest]
fn matrix_provider_from_parquet() {
    let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let bytes = write_parquet(array);
    let provider = DenseMatrixProvider::try_from_parquet_reader("demo", bytes, "features")
        .expect("parquet load");
    assert_eq!(provider.len(), 2);
    assert_eq!(provider.dimension(), 3);
    assert_eq!(provider.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[rstest]
fn matrix_provider_parquet_missing_column() {
    let array = build_array(&[[1.0, 2.0, 3.0]]);
    let bytes = write_parquet(array);
    let err = DenseMatrixProvider::try_from_parquet_reader("demo", bytes, "unknown")
        .expect_err("missing column");
    assert!(matches!(
        err,
        DenseMatrixProviderError::ColumnNotFound { column } if column == "unknown"
    ));
}

#[rstest]
fn matrix_provider_parquet_wrong_type() {
    let field = Field::new("features", DataType::Int32, false);
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3])) as ArrayRef],
    )
    .expect("batch");
    let mut buffer = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
    }
    let bytes = Bytes::from(buffer);
    let err = DenseMatrixProvider::try_from_parquet_reader("demo", bytes, "features")
        .expect_err("wrong type");
    assert!(matches!(
        err,
        DenseMatrixProviderError::InvalidColumnType { .. }
    ));
}
