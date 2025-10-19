//! Error types produced by the CPU HNSW implementation.

use thiserror::Error;

use crate::error::DataSourceError;

/// Errors produced by the CPU HNSW implementation.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum HnswError {
    /// Construction was attempted on an empty data source.
    #[error("cannot build an HNSW index from an empty data source")]
    EmptyBuild,
    /// Parameters were invalid for the current configuration.
    #[error("invalid HNSW parameter: {reason}")]
    InvalidParameters { reason: String },
    /// The same node was inserted more than once.
    #[error("node {node} has already been inserted")]
    DuplicateNode { node: usize },
    /// The graph is missing an entry point, which indicates a logic error.
    #[error("HNSW graph has no entry point")]
    GraphEmpty,
    /// Attempted to operate on an inconsistent graph state.
    #[error("HNSW graph invariant violated: {message}")]
    GraphInvariantViolation { message: String },
    /// The data source returned a non-finite distance.
    #[error("data source returned a non-finite distance for ({left}, {right})")]
    NonFiniteDistance { left: usize, right: usize },
    /// Wrapped [`DataSource`] error.
    #[error("data source failure: {0}")]
    DataSource(#[from] DataSourceError),
}
