//! A failure-injecting data source for one-shot HNSW error integration tests.

use std::sync::atomic::{AtomicBool, Ordering};

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
    data: Vec<f32>,
    should_fail: AtomicBool,
    mode: FailureMode,
}

impl FailableSource {
    #[must_use]
    pub fn failing(mode: FailureMode) -> Self {
        Self {
            data: vec![0.0, 1.0, 2.0],
            should_fail: AtomicBool::new(true),
            mode,
        }
    }
}

impl DataSource for FailableSource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        "failable-one-shot-source"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        if self.should_fail.load(Ordering::SeqCst) {
            return match self.mode {
                FailureMode::DataSource => Err(DataSourceError::OutOfBounds { index: i.max(j) }),
                FailureMode::NonFinite => Ok(f32::NAN),
            };
        }

        let left = self
            .data
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self
            .data
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((left - right).abs())
    }
}
