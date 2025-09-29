use super::{DenseMatrixProvider, DenseMatrixProviderError, support::*};
use crate::ingest::{copy_list_values, validate_fixed_size_list_field};
use chutoro_core::DataSource;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use rstest::rstest;
use std::sync::Arc;

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
        vec![Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3])) as _],
    )
    .expect("batch");
    let mut buffer = Vec::new();
    {
        let mut writer =
            parquet::arrow::arrow_writer::ArrowWriter::try_new(&mut buffer, schema, None)
                .expect("writer");
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
            vec![Arc::new(array) as _],
        )
        .expect("batch one")
    };
    let batch_two = {
        let rows = vec![vec![4.0, 5.0]];
        let array = build_list_array(&rows, 2, false);
        let field = feature_field(2, false, false);
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![field])),
            vec![Arc::new(array) as _],
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

#[test]
fn validate_field_rejects_negative_dimension() {
    let child = Arc::new(Field::new("item", DataType::Float32, false));
    let field = Field::new("features", DataType::FixedSizeList(child, -1), false);
    let err = validate_fixed_size_list_field(&field, "features")
        .expect_err("negative dimension must be rejected");
    assert!(matches!(
        err,
        DenseMatrixProviderError::InvalidDimension { actual } if actual == -1
    ));
}

#[test]
fn copy_list_values_rejects_incorrect_length() {
    let rows = vec![vec![1.0, 2.0]];
    let array = build_list_array(&rows, 2, false);
    let mut values = Vec::new();
    let err = copy_list_values(&array, 3, 0, &mut values)
        .expect_err("incorrect lengths must be rejected");
    assert!(matches!(
        err,
        DenseMatrixProviderError::InvalidRowLength {
            row: 0,
            expected: 3,
            actual: 2
        }
    ));
}
