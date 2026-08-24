//! Deliberately ambient path opening for the Parquet convenience constructor.
//!
//! This module is the sole dense-provider adapter that accepts a caller-supplied
//! filesystem path. Callers that already hold a capability or another readable
//! source should use `DenseMatrixProvider::try_from_parquet_reader` instead.

use std::{fs::File, path::Path};

use crate::errors::DenseMatrixProviderError;

/// Opens a caller-supplied Parquet path for the convenience constructor.
///
/// # Errors
///
/// Returns an error when the supplied path cannot be opened for reading.
pub(crate) fn open(path: impl AsRef<Path>) -> Result<File, DenseMatrixProviderError> {
    Ok(File::open(path)?)
}
