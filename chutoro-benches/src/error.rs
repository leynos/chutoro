//! Benchmark setup error type.
//!
//! Aggregates the various error types that may arise during benchmark
//! data preparation so that setup functions can propagate failures
//! with `?` instead of using `.expect()`.

use crate::profiling::ProfilingError;
use crate::source::SyntheticError;
use chutoro_core::{HierarchyError, HnswError, MstError};

/// Errors that may occur during benchmark setup.
#[derive(Debug, thiserror::Error)]
pub enum BenchSetupError {
    /// Synthetic data generation failed.
    #[error("synthetic source generation failed: {0}")]
    Synthetic(#[from] SyntheticError),
    /// HNSW parameter validation or build failed.
    #[error("HNSW operation failed: {0}")]
    Hnsw(#[from] HnswError),
    /// MST computation failed.
    #[error("MST computation failed: {0}")]
    Mst(#[from] MstError),
    /// Hierarchy extraction failed.
    #[error("hierarchy extraction failed: {0}")]
    Hierarchy(#[from] HierarchyError),
    /// A zero value was passed where a non-zero integer was required.
    #[error("expected a non-zero value for {context}")]
    ZeroValue {
        /// A description of the parameter that was unexpectedly zero.
        context: &'static str,
    },
    /// Memory profiling failed.
    #[error("memory profiling failed: {0}")]
    Profiling(#[from] ProfilingError),
    /// A data-source distance computation failed.
    #[error("data source error: {0}")]
    DataSource(#[from] chutoro_core::DataSourceError),
}
