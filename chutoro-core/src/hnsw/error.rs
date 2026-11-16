//! Error types produced by the CPU HNSW implementation.

use std::{fmt, sync::Arc};

use thiserror::Error;

use crate::error::{ChutoroError, DataSourceError, define_error_codes};

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
    /// A synchronization primitive became poisoned after a panic.
    #[error("lock for {resource} is poisoned")]
    LockPoisoned {
        /// Name of the locked resource that was poisoned.
        resource: &'static str,
    },
    /// Wrapped [`crate::DataSource`] error.
    #[error("data source failure: {0}")]
    DataSource(#[from] DataSourceError),
}

impl HnswError {
    /// Converts this HNSW adapter error into the public core error type.
    #[must_use]
    pub(crate) fn into_chutoro_error(self, data_source: Arc<str>) -> ChutoroError {
        match self {
            Self::DataSource(error) => ChutoroError::DataSource { data_source, error },
            other => ChutoroError::CpuHnswFailure {
                code: Arc::from(other.code().as_str()),
                message: Arc::from(other.to_string()),
            },
        }
    }
}

define_error_codes! {
    /// Machine-readable error codes for [`HnswError`].
    enum HnswErrorCode for HnswError {
        /// Construction was attempted on an empty data source.
        EmptyBuild => EmptyBuild => "EMPTY_BUILD",
        /// Parameters were invalid for the current configuration.
        InvalidParameters => InvalidParameters { .. } => "INVALID_PARAMETERS",
        /// The same node was inserted more than once.
        DuplicateNode => DuplicateNode { .. } => "DUPLICATE_NODE",
        /// The graph is missing an entry point, which indicates a logic error.
        GraphEmpty => GraphEmpty => "GRAPH_EMPTY",
        /// Attempted to operate on an inconsistent graph state.
        GraphInvariantViolation => GraphInvariantViolation { .. } => "GRAPH_INVARIANT_VIOLATION",
        /// The data source returned a non-finite distance.
        NonFiniteDistance => NonFiniteDistance { .. } => "NON_FINITE_DISTANCE",
        /// A synchronization primitive became poisoned after a panic.
        LockPoisoned => LockPoisoned { .. } => "LOCK_POISONED",
        /// Wrapped [`crate::DataSource`] error.
        DataSource => DataSource(_) => "DATA_SOURCE",
    }
}
