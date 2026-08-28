//! A failure-injecting data source for one-shot HNSW error integration tests.

use chutoro_core::{DataSource, DataSourceError};

/// Failure modes used to exercise HNSW error translation through public APIs.
#[derive(Clone, Copy, Debug)]
pub enum FailureMode {
    /// Return a data-source error from distance queries.
    DataSource,
    /// Return a non-finite distance that the HNSW adapter rejects.
    NonFinite,
}

/// A scalar data source whose distance queries can be made to fail.
pub struct FailableSource {
    mode: FailureMode,
}

impl FailableSource {
    #[must_use]
    pub const fn failing(mode: FailureMode) -> Self {
        Self { mode }
    }
}

impl DataSource for FailableSource {
    fn len(&self) -> usize {
        3
    }

    fn name(&self) -> &'static str {
        "failable-one-shot-source"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        match self.mode {
            FailureMode::DataSource => Err(DataSourceError::OutOfBounds { index: i.max(j) }),
            FailureMode::NonFinite => Ok(f32::NAN),
        }
    }
}
