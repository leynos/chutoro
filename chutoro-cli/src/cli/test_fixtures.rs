//! Test fixture builders for CLI integration tests.
//!
//! These helpers create small, representative inputs (e.g. Parquet files) used
//! across CLI tests. Keeping them in one place avoids duplication and keeps the
//! individual test modules focused on behaviour.

use std::fs::File;
use std::path::PathBuf;

use chutoro_providers_dense::testing::{
    build_list_array, feature_field, write_parquet_file as write_shared_parquet_file,
};
use tempfile::TempDir;

/// Creates a small Parquet file containing a fixed `features` column.
///
/// The shared provider fixture builders construct the `features` field and
/// two-dimensional array; this wrapper supplies the temporary path.
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
    let rows = vec![
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];
    let array = build_list_array(&rows, 2, false)?;
    let field = feature_field(2, false, false)?;
    let mut file = File::create(&path)?;
    write_shared_parquet_file(&mut file, field, array)?;
    Ok(path)
}
