//! Tests covering dense matrix ingestion from Arrow and Parquet sources.
use super::{DenseMatrixProvider, DenseMatrixProviderError, DenseSource};
use crate::provider::try_from_record_batches;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use chutoro_core::DataSource;
use parquet::arrow::arrow_writer::ArrowWriter;
use rstest::rstest;
use std::convert::TryFrom;
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
    let rows = rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    build_list_array(&rows, 3, false)
}

fn build_list_array(
    rows: &[Vec<f32>],
    dimension: usize,
    child_nullable: bool,
) -> FixedSizeListArray {
    assert!(rows.iter().all(|row| row.len() == dimension));
    let values = Float32Array::from_iter_values(rows.iter().flatten().copied());
    FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, child_nullable)),
        i32::try_from(dimension).expect("dimension fits in i32"),
        Arc::new(values) as ArrayRef,
        None,
    )
}

fn feature_field(
    dimension: usize,
    child_nullable: bool,
    list_nullable: bool,
) -> Field {
    Field::new(
        "features",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, child_nullable)),
            i32::try_from(dimension).expect("dimension fits in i32"),
        ),
        list_nullable,
    )
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
    let field = feature_field(3, false, false);
    write_parquet_with_field(field, array)
}

fn write_parquet_with_field(field: Field, array: FixedSizeListArray) -> Bytes {
    let schema = Arc::new(Schema::new(vec![field.clone()]));
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

#[rstest]
fn matrix_provider_parquet_inconsistent_dimension() {
    let batch_one = {
        let rows = vec![vec![1.0, 2.0, 3.0]];
        let array = build_list_array(&rows, 3, false);
        let field = feature_field(3, false, false);
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![field])),
            vec![Arc::new(array) as ArrayRef],
        )
        .expect("batch one")
    };
    let batch_two = {
        let rows = vec![vec![4.0, 5.0]];
        let array = build_list_array(&rows, 2, false);
        let field = feature_field(2, false, false);
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![field])),
            vec![Arc::new(array) as ArrayRef],
        )
        .expect("batch two")
    };
    let err = try_from_record_batches("demo", "features", vec![batch_one, batch_two])
        .expect_err("dimension mismatch must fail");
    assert!(matches!(
        err,
        DenseMatrixProviderError::InconsistentBatchDimension {
            expected: 3,
            actual: 2
        }
    ));
}

#[rstest]
#[case(true, false)]
#[case(false, true)]
fn matrix_provider_parquet_nullable_schema(
    #[case] list_nullable: bool,
    #[case] child_nullable: bool,
) {
    let rows = vec![vec![1.0, 2.0, 3.0]];
    let array = build_list_array(&rows, 3, child_nullable);
    let field = feature_field(3, child_nullable, list_nullable);
    let bytes = write_parquet_with_field(field, array);
    let err = DenseMatrixProvider::try_from_parquet_reader("demo", bytes, "features")
        .expect_err("nullable schema must be rejected");
    assert!(matches!(
        err,
        DenseMatrixProviderError::NullableField {
            column,
            nullable_child
        } if column == "features" && nullable_child == child_nullable
    ));
}
