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
    InvalidParameters {
        /// Human-readable explanation of the parameter failure.
        reason: String,
    },
    /// The same node was inserted more than once.
    #[error("node {node} has already been inserted")]
    DuplicateNode {
        /// Identifier of the node that was inserted repeatedly.
        node: usize,
    },
    /// The graph is missing an entry point, which indicates a logic error.
    #[error("HNSW graph has no entry point")]
    GraphEmpty,
    /// Attempted to operate on an inconsistent graph state.
    #[error("HNSW graph invariant violated: {message}")]
    GraphInvariantViolation {
        /// Description of the violated invariant to assist debugging.
        message: String,
    },
    /// The data source returned a non-finite distance.
    #[error("data source returned a non-finite distance for ({left}, {right})")]
    NonFiniteDistance {
        /// Index of the first node involved in the distance query.
        left: usize,
        /// Index of the second node involved in the distance query.
        right: usize,
    },
    /// Wrapped [`crate::DataSource`] error.
    #[error("data source failure: {0}")]
    DataSource(#[from] DataSourceError),
}

impl HnswError {
    /// Returns a stable, machine-readable error code for the variant.
    #[must_use]
    pub const fn code(&self) -> HnswErrorCode {
        match self {
            Self::EmptyBuild => HnswErrorCode::EmptyBuild,
            Self::InvalidParameters { .. } => HnswErrorCode::InvalidParameters,
            Self::DuplicateNode { .. } => HnswErrorCode::DuplicateNode,
            Self::GraphEmpty => HnswErrorCode::GraphEmpty,
            Self::GraphInvariantViolation { .. } => HnswErrorCode::GraphInvariantViolation,
            Self::NonFiniteDistance { .. } => HnswErrorCode::NonFiniteDistance,
            Self::DataSource(_) => HnswErrorCode::DataSource,
        }
    }
}

/// Machine-readable error codes for [`HnswError`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HnswErrorCode {
    /// Construction was attempted on an empty data source.
    EmptyBuild,
    /// Parameters were invalid for the current configuration.
    InvalidParameters,
    /// The same node was inserted more than once.
    DuplicateNode,
    /// The graph is missing an entry point, which indicates a logic error.
    GraphEmpty,
    /// Attempted to operate on an inconsistent graph state.
    GraphInvariantViolation,
    /// The data source returned a non-finite distance.
    NonFiniteDistance,
    /// Wrapped [`crate::DataSource`] error.
    DataSource,
}

impl HnswErrorCode {
    /// Returns the symbolic identifier for logging and metrics surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyBuild => "EMPTY_BUILD",
            Self::InvalidParameters => "INVALID_PARAMETERS",
            Self::DuplicateNode => "DUPLICATE_NODE",
            Self::GraphEmpty => "GRAPH_EMPTY",
            Self::GraphInvariantViolation => "GRAPH_INVARIANT_VIOLATION",
            Self::NonFiniteDistance => "NON_FINITE_DISTANCE",
            Self::DataSource => "DATA_SOURCE",
        }
    }
}
