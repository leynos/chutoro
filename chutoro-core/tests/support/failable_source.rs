//! A failure-injecting data source shared by HNSW error tests.

use std::sync::atomic::{AtomicBool, Ordering};

use super::{DataSource, DataSourceError, MetricDescriptor};

/// Failure modes used to exercise HNSW error translation through public APIs.
#[derive(Clone, Copy, Debug)]
pub enum FailureMode {
    /// Return a data-source error from distance queries.
    DataSource,
    /// Return a non-finite distance that the HNSW adapter rejects.
    NonFinite,
    /// Fail only the given unordered pair after failure injection is enabled.
    PairDataSource {
        /// First source index in the failing pair.
        left: usize,
        /// Second source index in the failing pair.
        right: usize,
    },
}

/// A scalar data source whose distance queries can be made to fail or recover.
pub struct FailableSource {
    values: Vec<f32>,
    should_fail: AtomicBool,
    mode: FailureMode,
}

impl FailableSource {
    /// Constructs a working source configured for a particular failure mode.
    #[must_use]
    pub fn new(mode: FailureMode) -> Self {
        Self {
            values: vec![0.0, 1.0, 2.0],
            should_fail: AtomicBool::new(false),
            mode,
        }
    }

    /// Enables the configured distance-query failure.
    pub fn fail(&self) {
        self.should_fail.store(true, Ordering::SeqCst);
    }

    /// Restores successful distance queries for the configured source values.
    pub fn recover(&self) {
        self.should_fail.store(false, Ordering::SeqCst);
    }
}

impl DataSource for FailableSource {
    /// Returns the number of scalar values served by this fixture.
    fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns a stable name used in mapped data-source errors.
    fn name(&self) -> &'static str {
        "failable-source"
    }

    /// Returns the configured injected failure or the absolute scalar distance.
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        if self.should_fail.load(Ordering::SeqCst) {
            return match self.mode {
                FailureMode::DataSource => Err(DataSourceError::OutOfBounds { index: i.max(j) }),
                FailureMode::NonFinite => Ok(f32::NAN),
                FailureMode::PairDataSource { left, right } if is_pair(i, j, left, right) => {
                    Err(DataSourceError::OutOfBounds { index: i.max(j) })
                }
                FailureMode::PairDataSource { .. } => self.distance_for_values(i, j),
            };
        }

        self.distance_for_values(i, j)
    }

    /// Describes the fixture's absolute-distance metric.
    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("failable-source:abs")
    }
}

impl FailableSource {
    /// Calculates the absolute distance between two valid fixture values.
    fn distance_for_values(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let left = self
            .values
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self
            .values
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok(left.mul_add(1.0, std::ops::Neg::neg(*right)).abs())
    }
}

/// Returns whether two indices match an unordered configured pair.
const fn is_pair(i: usize, j: usize, left: usize, right: usize) -> bool {
    (i == left && j == right) || (i == right && j == left)
}
