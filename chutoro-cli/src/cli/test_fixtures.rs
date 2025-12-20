//! Test fixture builders for CLI integration tests.
//!
//! These helpers create small, representative inputs (e.g. Parquet files) used
//! across CLI tests. Keeping them in one place avoids duplication and keeps the
//! individual test modules focused on behaviour.

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use tempfile::TempDir;

/// Creates a small Parquet file containing a fixed `features` column.
///
/// The file is written to `dir` using the provided `name`, and contains a
/// single record batch with a `features: FixedSizeList<Float32, 2>` column.
///
/// This is intended for CLI tests that exercise Parquet ingestion without
/// relying on external fixtures.
///
/// # Errors
/// Returns an error when the file cannot be created or the Parquet writer fails
/// to write the batch.
pub fn create_parquet_file(
    dir: &TempDir,
    name: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = dir.path().join(name);
    let schema = build_schema();
    let batch = build_record_batch(schema.clone());
    let file = File::create(&path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(path)
}

fn build_schema() -> Arc<Schema> {
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let list_type = DataType::FixedSizeList(item_field.clone(), 2);
    Arc::new(Schema::new(vec![Field::new("features", list_type, false)]))
}

fn build_record_batch(schema: Arc<Schema>) -> RecordBatch {
    // Flat buffer representing 4 2D points: (0,0), (1,1), (2,2), (3,3).
    let values = Float32Array::from(vec![0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let list = FixedSizeListArray::new(item_field, 2, Arc::new(values) as ArrayRef, None);
    RecordBatch::try_new(schema, vec![Arc::new(list) as ArrayRef])
        .expect("failed to construct record batch")
}
