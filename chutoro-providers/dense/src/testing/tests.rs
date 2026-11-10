//! Contract tests for the shared Arrow array and Parquet fixture builders.

use std::{io::Read, sync::Arc};

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field};
use bytes::Bytes;
use cap_std::fs::Dir;
use parquet::{arrow::arrow_reader::ParquetRecordBatchReaderBuilder, file::reader::ChunkReader};
use tempfile::tempdir;

use super::*;

#[test]
fn plain_builder_preserves_dimension_rows_and_child_nullability() {
    let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let array = build_list_array(&rows, 2, true).expect("fixture array must build");

    assert_eq!(array.len(), 2);
    assert_eq!(array.value_length(), 2);
    assert_eq!(array.null_count(), 0);
    assert_eq!(
        array.data_type(),
        &DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2)
    );
    let values = array
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("child values must be Float32");
    assert_eq!(values.value(0).to_bits(), 1.0_f32.to_bits());
    assert_eq!(values.value(1).to_bits(), 2.0_f32.to_bits());
    assert_eq!(values.value(2).to_bits(), 3.0_f32.to_bits());
    assert_eq!(values.value(3).to_bits(), 4.0_f32.to_bits());
}

#[test]
fn const_generic_builder_uses_the_rows_dimension() {
    let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).expect("fixture array must build");

    assert_eq!(array.len(), 2);
    assert_eq!(array.value_length(), 3);
    assert_eq!(array.null_count(), 0);
}

#[test]
fn row_null_builder_marks_only_the_requested_rows() {
    let rows = vec![Some(vec![1.0, 2.0]), None, Some(vec![3.0, 4.0])];
    let array = build_list_array_with_row_nulls(&rows, 2).expect("fixture array must build");

    assert_eq!(array.len(), 3);
    assert_eq!(array.value_length(), 2);
    assert_eq!(array.null_count(), 1);
    assert!(!array.is_null(0));
    assert!(array.is_null(1));
    assert!(!array.is_null(2));
    assert_eq!(
        array.data_type(),
        &DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 2)
    );
}

#[test]
fn value_null_builder_marks_only_the_requested_values() {
    let rows = vec![vec![Some(1.0), None], vec![Some(3.0), Some(4.0)]];
    let array = build_list_array_with_value_nulls(&rows, 2).expect("fixture array must build");
    let values = array
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("child values must be Float32");

    assert_eq!(array.len(), 2);
    assert_eq!(array.value_length(), 2);
    assert_eq!(array.null_count(), 0);
    assert_eq!(values.null_count(), 1);
    assert!(values.is_null(1));
    assert_eq!(
        array.data_type(),
        &DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2)
    );
}

#[test]
fn feature_field_keeps_item_and_list_nullability_independent() {
    let field = feature_field(4, true, false).expect("feature field must build");
    let DataType::FixedSizeList(item, dimension) = field.data_type() else {
        panic!("feature field must be a fixed-size list");
    };

    assert_eq!(*dimension, 4);
    assert!(item.is_nullable());
    assert!(!field.is_nullable());
}

#[test]
fn list_builder_reports_the_first_row_with_the_wrong_width() {
    let rows = vec![vec![1.0, 2.0], vec![3.0]];

    assert!(matches!(
        build_list_array(&rows, 2, false),
        Err(FixtureError::RowLength {
            row: 1,
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn bytes_and_file_writers_preserve_equivalent_record_batches() {
    let rows = vec![vec![Some(1.0), None], vec![Some(3.0), Some(4.0)]];
    let array = build_list_array_with_value_nulls(&rows, 2).expect("fixture array must build");
    let field = feature_field(2, true, false).expect("feature field must build");
    let bytes = write_parquet_with_field(field.clone(), array.clone())
        .expect("in-memory Parquet fixture must serialize");
    let bytes_batches =
        read_record_batches(bytes).expect("in-memory Parquet fixture must be readable");

    let temporary_directory = tempdir().expect("temporary directory must be created");
    let directory = Dir::open_ambient_dir(temporary_directory.path(), cap_std::ambient_authority())
        .expect("temporary directory must be opened as a capability");
    let mut output = directory
        .create("fixture.parquet")
        .expect("fixture file must be created");
    write_parquet_file(&mut output, field, array).expect("file Parquet fixture must serialize");
    drop(output);
    let mut file = directory
        .open("fixture.parquet")
        .expect("fixture file must be readable");
    let mut file_bytes = Vec::new();
    file.read_to_end(&mut file_bytes)
        .expect("fixture bytes must be readable");
    let file_batches = read_record_batches(Bytes::from(file_bytes))
        .expect("file Parquet fixture must be readable");

    assert_eq!(bytes_batches.len(), 1);
    assert_eq!(file_batches.len(), 1);
    let bytes_batch = bytes_batches
        .first()
        .expect("in-memory fixture must contain a record batch");
    let file_batch = file_batches
        .first()
        .expect("file fixture must contain a record batch");
    assert_record_batches_equal(bytes_batch, file_batch);
}

fn read_record_batches<R: ChunkReader + 'static>(
    input: R,
) -> Result<Vec<RecordBatch>, FixtureError> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?.build()?;
    Ok(reader.collect::<Result<Vec<_>, _>>()?)
}

fn assert_record_batches_equal(expected: &RecordBatch, actual: &RecordBatch) {
    assert_eq!(expected.schema(), actual.schema());
    assert_eq!(expected.num_rows(), actual.num_rows());
    assert_eq!(expected.num_columns(), actual.num_columns());

    let expected_list = expected
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("feature column must be FixedSizeList");
    let actual_list = actual
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("feature column must be FixedSizeList");
    assert_eq!(expected_list.data_type(), actual_list.data_type());
    assert_eq!(expected_list.null_count(), actual_list.null_count());

    let expected_values = expected_list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("expected child values must be Float32");
    let actual_values = actual_list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("actual child values must be Float32");
    assert_eq!(expected_values.null_count(), actual_values.null_count());
    for index in 0..expected_values.len() {
        assert_eq!(expected_values.is_null(index), actual_values.is_null(index));
        if !expected_values.is_null(index) {
            assert_eq!(
                expected_values.value(index).to_bits(),
                actual_values.value(index).to_bits()
            );
        }
    }
}
