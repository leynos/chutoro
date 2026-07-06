//! Shared fixtures for `chutoro-core` integration tests.
//!
//! This module exports the `Dummy` `DataSource` fixture used by integration
//! tests to supply small in-memory scalar datasets with deterministic absolute
//! distance semantics. It keeps common test scaffolding beside the integration
//! test crates that exercise the public `chutoro-core` API.

use chutoro_core::{DataSource, DataSourceError};

#[derive(Clone)]
pub struct Dummy {
    data: Vec<f32>,
}

impl Dummy {
    #[must_use]
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl DataSource for Dummy {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        "dummy"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self
            .data
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .data
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }
}
