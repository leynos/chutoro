use super::{DenseMatrixProvider, DenseMatrixProviderError};
use crate::ingest::{append_fixed_size_list_values, validate_fixed_size_list_field};
use arrow_array::builder::BooleanBufferBuilder;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use parquet::arrow::arrow_writer::ArrowWriter;
use std::{convert::TryFrom, iter, sync::Arc};

pub(crate) fn build_array(rows: &[[f32; 3]]) -> FixedSizeListArray {
    let rows = rows.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
    build_list_array(&rows, 3, false)
}

pub(crate) fn build_list_array(
    rows: &[Vec<f32>],
    dimension: usize,
    child_nullable: bool,
) -> FixedSizeListArray {
    assert!(rows.iter().all(|row| row.len() == dimension));
    let values = Float32Array::from_iter_values(rows.iter().flatten().copied());
    fixed_size_list_from_values(values, dimension, child_nullable)
}

pub(crate) fn build_list_array_with_row_nulls(
    rows: &[Option<Vec<f32>>],
    dimension: usize,
) -> FixedSizeListArray {
    assert!(
        rows.iter()
            .all(|row| { row.as_ref().is_none_or(|values| values.len() == dimension) })
    );
    let mut flat = Vec::with_capacity(rows.len() * dimension);
    let mut validity = BooleanBufferBuilder::new(rows.len());
    for row in rows {
        match row {
            Some(values) => {
                validity.append(true);
                flat.extend_from_slice(values);
            }
            None => {
                validity.append(false);
                flat.extend(iter::repeat_n(0.0, dimension));
            }
        }
    }
    let values = Float32Array::from(flat);
    FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        i32::try_from(dimension).expect("dimension fits in i32"),
        Arc::new(values) as ArrayRef,
        Some(validity.finish().into()),
    )
}

pub(crate) fn build_list_array_with_value_nulls(
    rows: &[Vec<Option<f32>>],
    dimension: usize,
) -> FixedSizeListArray {
    assert!(rows.iter().all(|row| row.len() == dimension));
    let values = Float32Array::from_iter(rows.iter().flatten().copied());
    fixed_size_list_from_values(values, dimension, true)
}

pub(crate) fn feature_field(dimension: usize, child_nullable: bool, list_nullable: bool) -> Field {
    Field::new(
        "features",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, child_nullable)),
            i32::try_from(dimension).expect("dimension fits in i32"),
        ),
        list_nullable,
    )
}

pub(crate) fn try_from_record_batches(
    name: impl Into<String>,
    column: &str,
    batches: Vec<RecordBatch>,
) -> Result<DenseMatrixProvider, DenseMatrixProviderError> {
    let mut values = Vec::new();
    let mut rows = 0_usize;
    let mut dimension: Option<usize> = None;

    for batch in batches {
        let schema = batch.schema();
        let index =
            schema
                .index_of(column)
                .map_err(|_| DenseMatrixProviderError::ColumnNotFound {
                    column: column.to_owned(),
                })?;
        let field = schema.field(index);
        let width = validate_fixed_size_list_field(field, column)?;
        if let Some(expected) = dimension {
            if expected != width {
                return Err(DenseMatrixProviderError::InconsistentBatchDimension {
                    expected,
                    actual: width,
                });
            }
        } else {
            dimension = Some(width);
        }
        let column_array = batch.column(index);
        let list = column_array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| DenseMatrixProviderError::InvalidColumnType {
                column: column.to_owned(),
                actual: column_array.data_type().clone(),
            })?;
        append_fixed_size_list_values(list, dimension, rows, &mut values)?;
        rows += list.len();
    }

    let dimension = dimension.unwrap_or(0);
    Ok(DenseMatrixProvider::from_parts(
        name, rows, dimension, values,
    ))
}

pub(crate) fn write_parquet(array: FixedSizeListArray) -> Bytes {
    let field = feature_field(3, false, false);
    write_parquet_with_field(field, array)
}

pub(crate) fn write_parquet_with_field(field: Field, array: FixedSizeListArray) -> Bytes {
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

pub(crate) fn write_parquet_two_batches(
    first: FixedSizeListArray,
    second: FixedSizeListArray,
    field: Field,
) -> Bytes {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch_one =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(first) as ArrayRef]).expect("batch one");
    let batch_two = RecordBatch::try_new(schema.clone(), vec![Arc::new(second) as ArrayRef])
        .expect("batch two");
    let mut buffer = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(&mut buffer, schema, None).expect("writer");
        writer.write(&batch_one).expect("write batch one");
        writer.write(&batch_two).expect("write batch two");
        writer.close().expect("close");
    }
    Bytes::from(buffer)
}

fn fixed_size_list_from_values(
    values: Float32Array,
    dimension: usize,
    child_nullable: bool,
) -> FixedSizeListArray {
    FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, child_nullable)),
        i32::try_from(dimension).expect("dimension fits in i32"),
        Arc::new(values) as ArrayRef,
        None,
    )
}

