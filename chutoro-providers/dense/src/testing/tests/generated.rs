//! Generated contracts for shared dense fixture builders.

use std::{io::Read, sync::Arc};

use arrow_array::{Array, FixedSizeListArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use cap_std::fs::Dir;
use proptest::prelude::*;
use tempfile::tempdir;

use super::super::{
    build_list_array, build_list_array_with_row_nulls, build_list_array_with_value_nulls,
    feature_field, write_parquet_file, write_parquet_with_field,
};
use super::{assert_record_batches_equal, child_values, read_record_batches};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn generated_dimensions_rows_values_and_null_patterns_survive_both_writers(
        dimension in 1usize..=8,
        row_count in 1usize..=12,
        raw_values in prop::collection::vec(-1000i16..=1000, 96),
        row_null_pattern in prop::collection::vec(any::<bool>(), 12),
        value_null_pattern in prop::collection::vec(any::<bool>(), 96),
    ) {
        let value_count = dimension * row_count;
        let values = raw_values
            .iter()
            .take(value_count)
            .map(|value| f32::from(*value))
            .collect::<Vec<_>>();
        let expected_width = i32::try_from(dimension)
            .expect("generated dimensions must fit Arrow's list width");
        let plain_rows = values
            .chunks(dimension)
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();

        let plain_array = build_list_array(&plain_rows, dimension, true)
            .expect("generated dense rows must build");
        prop_assert_eq!(plain_array.len(), row_count);
        prop_assert_eq!(plain_array.value_length(), expected_width);
        prop_assert_eq!(plain_array.null_count(), 0);
        prop_assert_eq!(
            plain_array.data_type(),
            &DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                expected_width,
            ),
        );
        prop_assert_eq!(
            child_values(&plain_array)
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        );

        let row_null_rows = plain_rows
            .iter()
            .zip(row_null_pattern.iter())
            .map(|(row, is_null)| {
                if *is_null {
                    None
                } else {
                    Some(row.clone())
                }
            })
            .collect::<Vec<_>>();
        let row_null_array = build_list_array_with_row_nulls(&row_null_rows, dimension)
            .expect("generated row-null rows must build");
        prop_assert_eq!(row_null_array.len(), row_count);
        prop_assert_eq!(row_null_array.value_length(), expected_width);
        prop_assert_eq!(
            row_null_array.null_count(),
            row_null_pattern
                .iter()
                .take(row_count)
                .filter(|is_null| **is_null)
                .count(),
        );
        prop_assert_eq!(
            row_null_array.data_type(),
            &DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                expected_width,
            ),
        );
        prop_assert_eq!(
            (0..row_count).map(|index| row_null_array.is_null(index)).collect::<Vec<_>>(),
            row_null_pattern
                .iter()
                .take(row_count)
                .copied()
                .collect::<Vec<_>>(),
        );
        let expected_row_values = plain_rows
            .iter()
            .zip(row_null_pattern.iter())
            .flat_map(|(row, is_null)| {
                row.iter().map(move |value| {
                    if *is_null {
                        0.0_f32
                    } else {
                        *value
                    }
                })
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(
            child_values(&row_null_array)
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_row_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        );

        let values_with_nulls = values
            .iter()
            .zip(value_null_pattern.iter())
            .map(|(value, is_null)| {
                if *is_null {
                    None
                } else {
                    Some(*value)
                }
            })
            .collect::<Vec<_>>();
        let value_null_rows = values_with_nulls
            .chunks(dimension)
            .map(<[Option<f32>]>::to_vec)
            .collect::<Vec<_>>();
        let value_null_array = build_list_array_with_value_nulls(&value_null_rows, dimension)
            .expect("generated value-null rows must build");
        prop_assert_eq!(value_null_array.len(), row_count);
        prop_assert_eq!(value_null_array.value_length(), expected_width);
        prop_assert_eq!(value_null_array.null_count(), 0);
        prop_assert_eq!(
            value_null_array.data_type(),
            &DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                expected_width,
            ),
        );
        let value_null_values = child_values(&value_null_array);
        for (index, (value, is_null)) in values
            .iter()
            .zip(value_null_pattern.iter())
            .enumerate()
        {
            prop_assert_eq!(value_null_values.is_null(index), *is_null);
            if !is_null {
                prop_assert_eq!(value_null_values.value(index).to_bits(), value.to_bits());
            }
        }

        assert_parquet_writers_preserve_array(
            feature_field(dimension, true, false)
                .expect("generated plain field must fit Arrow's list width"),
            plain_array,
        );
        assert_parquet_writers_preserve_array(
            feature_field(dimension, false, true)
                .expect("generated row-null field must fit Arrow's list width"),
            row_null_array,
        );
        assert_parquet_writers_preserve_array(
            feature_field(dimension, true, false)
                .expect("generated value-null field must fit Arrow's list width"),
            value_null_array,
        );
    }
}

/// Checks that both Parquet output forms preserve an input array and schema.
fn assert_parquet_writers_preserve_array(field: Field, array: FixedSizeListArray) {
    let schema = Arc::new(Schema::new(vec![field.clone()]));
    let expected = RecordBatch::try_new(schema, vec![Arc::new(array.clone())])
        .expect("fixture array must match its field");
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
    assert_record_batches_equal(&expected, bytes_batch);
    assert_record_batches_equal(&expected, file_batch);
    assert_record_batches_equal(bytes_batch, file_batch);
}
