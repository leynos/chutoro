//! Contract tests for the shared Arrow array and Parquet fixture builders.

use std::{io::Read, sync::Arc};

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use cap_std::fs::Dir;
use parquet::{arrow::arrow_reader::ParquetRecordBatchReaderBuilder, file::reader::ChunkReader};
use rstest::rstest;
use tempfile::tempdir;

use super::*;

#[path = "tests/generated.rs"]
mod generated;

#[test]
fn plain_builder_preserves_dimension_rows_and_child_nullability() {
    let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let array = build_list_array(&rows, 2, true).expect("fixture array must build");

    assert_list_shape(&array, 2, 2, 0);
    assert_list_type(&array, 2, true);
    assert_dense_child_values(&array, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn const_generic_builder_uses_the_rows_dimension() {
    let array = build_array(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).expect("fixture array must build");

    assert_eq!(array.len(), 2);
    assert_eq!(array.value_length(), 3);
    assert_eq!(array.null_count(), 0);
}

#[test]
fn list_builder_preserves_non_three_width_and_ordered_values() {
    let rows = [[9.0, 1.0, 5.0, 2.0], [-3.0, 7.0, 4.0, 8.0]];
    let array = build_array(&rows).expect("four-wide fixture must build");

    assert_list_shape(&array, 2, 4, 0);
    assert_list_type(&array, 4, false);
    assert_dense_child_values(&array, &[9.0, 1.0, 5.0, 2.0, -3.0, 7.0, 4.0, 8.0]);
}

#[test]
fn row_null_builder_marks_only_the_requested_rows() {
    let rows = vec![Some(vec![1.0, 2.0]), None, Some(vec![3.0, 4.0])];
    let array = build_list_array_with_row_nulls(&rows, 2).expect("fixture array must build");

    assert_list_shape(&array, 3, 2, 1);
    assert_null_pattern(&array, &[false, true, false]);
    assert_list_type(&array, 2, false);
    assert_dense_child_values(&array, &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
}

#[test]
fn value_null_builder_marks_only_the_requested_values() {
    let rows = vec![vec![Some(1.0), None], vec![Some(3.0), Some(4.0)]];
    let array = build_list_array_with_value_nulls(&rows, 2).expect("fixture array must build");
    assert_list_shape(&array, 2, 2, 0);
    assert_null_pattern(child_values(&array), &[false, true, false, false]);
    assert_list_type(&array, 2, true);
    assert_child_values_with_nulls(&array, &[Some(1.0), None, Some(3.0), Some(4.0)]);
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

#[rstest]
fn feature_field_reports_dimensions_larger_than_arrow_width() {
    let too_large = usize::try_from(i32::MAX)
        .expect("i32::MAX must fit the platform's usize")
        .saturating_add(1);

    assert!(matches!(
        feature_field(too_large, false, false),
        Err(FixtureError::Dimension(_))
    ));
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
    assert_value_null_batch_matches_expected(bytes_batch);
    assert_record_batches_equal(bytes_batch, file_batch);
}

/// Checks a decoded value-null fixture against an independent expected schema.
fn assert_value_null_batch_matches_expected(batch: &RecordBatch) {
    let expected_field = Field::new(
        "features",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
        false,
    );
    let expected_schema = Arc::new(Schema::new(vec![expected_field]));
    assert_eq!(batch.schema(), expected_schema);
    assert_eq!(batch.num_rows(), 2);

    let list = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("decoded feature column must be a fixed-size list");
    assert_list_shape(list, 2, 2, 0);
    assert_list_type(list, 2, true);
    assert_null_pattern(child_values(list), &[false, true, false, false]);
    assert_child_values_with_nulls(list, &[Some(1.0), None, Some(3.0), Some(4.0)]);
}

fn read_record_batches<R: ChunkReader + 'static>(
    input: R,
) -> Result<Vec<RecordBatch>, FixtureError> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?.build()?;
    Ok(reader.collect::<Result<Vec<_>, _>>()?)
}

/// Asserts decoded batches match in schema, validity, row count, and float bits.
fn assert_record_batches_equal(expected: &RecordBatch, actual: &RecordBatch) {
    assert_eq!(expected.schema(), actual.schema());
    assert_eq!(expected.num_rows(), actual.num_rows());
    assert_eq!(expected.num_columns(), actual.num_columns());

    let expected_list = feature_list(expected);
    let actual_list = feature_list(actual);
    assert_eq!(expected_list.data_type(), actual_list.data_type());
    assert_eq!(expected_list.null_count(), actual_list.null_count());
    assert_row_validity_matches(expected_list, actual_list);
    assert_child_rows_equal(expected_list, actual_list);
}

/// Downcasts the feature column to its fixed-size list representation.
fn feature_list(batch: &RecordBatch) -> &FixedSizeListArray {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("feature column must be FixedSizeList")
}

/// Asserts each row's outer validity matches.
fn assert_row_validity_matches(expected: &FixedSizeListArray, actual: &FixedSizeListArray) {
    for index in 0..expected.len() {
        assert_eq!(
            expected.is_null(index),
            actual.is_null(index),
            "row {index} validity differs"
        );
    }
}

/// Asserts every present row's child values match.
fn assert_child_rows_equal(expected: &FixedSizeListArray, actual: &FixedSizeListArray) {
    let present_rows = (0..expected.len()).filter(|index| !expected.is_null(*index));
    for row_index in present_rows {
        assert_child_row_equal(expected, actual, row_index);
    }
}

/// Asserts one present row's child values match in width, validity, and bits.
fn assert_child_row_equal(
    expected: &FixedSizeListArray,
    actual: &FixedSizeListArray,
    row_index: usize,
) {
    let expected_row = expected.value(row_index);
    let actual_row = actual.value(row_index);
    let expected_values = float_values(&expected_row);
    let actual_values = float_values(&actual_row);
    assert_eq!(
        expected_values.len(),
        actual_values.len(),
        "row {row_index} width differs"
    );
    for value_index in 0..expected_values.len() {
        assert_child_value_equal(expected_values, actual_values, row_index, value_index);
    }
}

/// Asserts one child value matches in validity and float bits.
fn assert_child_value_equal(
    expected: &Float32Array,
    actual: &Float32Array,
    row_index: usize,
    value_index: usize,
) {
    assert_eq!(
        expected.is_null(value_index),
        actual.is_null(value_index),
        "row {row_index} value {value_index} validity differs"
    );
    if !expected.is_null(value_index) {
        assert_eq!(
            expected.value(value_index).to_bits(),
            actual.value(value_index).to_bits(),
            "row {row_index} value {value_index} differs"
        );
    }
}

/// Downcasts one row's child values to their float representation.
fn float_values(array: &ArrayRef) -> &Float32Array {
    array
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("child values must be Float32")
}

fn assert_list_shape(
    array: &FixedSizeListArray,
    expected_rows: usize,
    expected_dimension: i32,
    expected_null_rows: usize,
) {
    assert_eq!(array.len(), expected_rows);
    assert_eq!(array.value_length(), expected_dimension);
    assert_eq!(array.null_count(), expected_null_rows);
}

fn assert_list_type(array: &FixedSizeListArray, dimension: i32, child_nullable: bool) {
    assert_eq!(
        array.data_type(),
        &DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, child_nullable)),
            dimension,
        )
    );
}

fn assert_null_pattern(array: &dyn Array, expected: &[bool]) {
    let actual = (0..array.len())
        .map(|index| array.is_null(index))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

fn assert_dense_child_values(array: &FixedSizeListArray, expected: &[f32]) {
    let values = child_values(array);
    assert_eq!(values.null_count(), 0);

    let expected_bits = expected
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let actual_bits = values
        .values()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(actual_bits, expected_bits);
}

/// Checks child validity and values against the expected nullable scalar rows.
fn assert_child_values_with_nulls(array: &FixedSizeListArray, expected: &[Option<f32>]) {
    let values = child_values(array);
    assert_eq!(values.len(), expected.len());
    for (index, expected_value) in expected.iter().enumerate() {
        match expected_value {
            Some(value) => {
                assert!(!values.is_null(index));
                assert_eq!(values.value(index).to_bits(), value.to_bits());
            }
            None => assert!(values.is_null(index)),
        }
    }
}

fn child_values(array: &FixedSizeListArray) -> &Float32Array {
    array
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("child values must be Float32")
}
