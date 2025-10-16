//! Error types for the Chutoro core library.
//!
//! Defines error enums exposed by the public API and a convenient result alias.

use std::{fmt, num::NonZeroUsize, sync::Arc};

use thiserror::Error;

use crate::builder::ExecutionStrategy;

/// Stable codes describing [`DataSourceError`] variants.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DataSourceErrorCode {
    /// Requested index was outside the source's bounds.
    OutOfBounds,
    /// Provided output buffer length did not match number of pairs.
    OutputLengthMismatch,
    /// Compared vectors had different dimensions.
    DimensionMismatch,
    /// Data source contained no rows.
    EmptyData,
    /// Data source rows must have positive dimension.
    ZeroDimension,
}

impl DataSourceErrorCode {
    /// Return the stable machine-readable representation of this error code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutOfBounds => "DATA_SOURCE_OUT_OF_BOUNDS",
            Self::OutputLengthMismatch => "DATA_SOURCE_OUTPUT_LENGTH_MISMATCH",
            Self::DimensionMismatch => "DATA_SOURCE_DIMENSION_MISMATCH",
            Self::EmptyData => "DATA_SOURCE_EMPTY",
            Self::ZeroDimension => "DATA_SOURCE_ZERO_DIMENSION",
        }
    }
}

impl fmt::Display for DataSourceErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An error produced by [`DataSource`] operations.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum DataSourceError {
    /// Requested index was outside the source's bounds.
    #[error("index {index} is out of bounds")]
    OutOfBounds { index: usize },
    /// Provided output buffer length did not match number of pairs.
    #[error("output buffer has length {out} but {expected} pairs were given")]
    OutputLengthMismatch { out: usize, expected: usize },
    /// Compared vectors had different dimensions.
    #[error("dimension mismatch: left={left}, right={right}")]
    DimensionMismatch { left: usize, right: usize },
    /// Data source contained no rows.
    #[error("data source contains no rows")]
    EmptyData,
    /// Data source rows must have positive dimension.
    #[error("data source vectors must have positive dimension")]
    ZeroDimension,
}

impl DataSourceError {
    /// Retrieve the stable [`DataSourceErrorCode`] for this error.
    pub const fn code(&self) -> DataSourceErrorCode {
        match self {
            Self::OutOfBounds { .. } => DataSourceErrorCode::OutOfBounds,
            Self::OutputLengthMismatch { .. } => DataSourceErrorCode::OutputLengthMismatch,
            Self::DimensionMismatch { .. } => DataSourceErrorCode::DimensionMismatch,
            Self::EmptyData => DataSourceErrorCode::EmptyData,
            Self::ZeroDimension => DataSourceErrorCode::ZeroDimension,
        }
    }
}

/// Stable codes describing [`ChutoroError`] variants.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ChutoroErrorCode {
    /// Minimum cluster size must be greater than zero.
    InvalidMinClusterSize,
    /// The supplied [`DataSource`] contained no items.
    EmptySource,
    /// The [`DataSource`] did not contain enough items for the configured
    /// minimum cluster size.
    InsufficientItems,
    /// The requested execution strategy is unavailable in the current build.
    BackendUnavailable,
    /// A [`DataSource`] operation failed while running the algorithm.
    DataSourceFailure,
}

impl ChutoroErrorCode {
    /// Return the stable machine-readable representation of this error code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidMinClusterSize => "CHUTORO_INVALID_MIN_CLUSTER_SIZE",
            Self::EmptySource => "CHUTORO_EMPTY_SOURCE",
            Self::InsufficientItems => "CHUTORO_INSUFFICIENT_ITEMS",
            Self::BackendUnavailable => "CHUTORO_BACKEND_UNAVAILABLE",
            Self::DataSourceFailure => "CHUTORO_DATA_SOURCE_FAILURE",
        }
    }
}

impl fmt::Display for ChutoroErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error type produced when constructing or running [`Chutoro`].
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ChutoroError {
    /// Minimum cluster size must be greater than zero.
    #[error("min_cluster_size must be at least 1 (got {got})")]
    InvalidMinClusterSize { got: usize },
    /// The supplied [`DataSource`] contained no items.
    #[error("data source `{data_source}` contains no items")]
    EmptySource { data_source: Arc<str> },
    /// The [`DataSource`] did not contain enough items for the configured
    /// `min_cluster_size`.
    #[error(
        "data source `{data_source}` has {items} items but min_cluster_size requires {min_cluster_size}"
    )]
    InsufficientItems {
        data_source: Arc<str>,
        items: usize,
        min_cluster_size: NonZeroUsize,
    },
    /// The requested execution strategy is unavailable in the current build.
    #[error("the requested execution strategy {requested:?} is not available in this build")]
    BackendUnavailable { requested: ExecutionStrategy },
    /// A [`DataSource`] operation failed while running the algorithm.
    #[error("data source `{data_source}` failed: {error}")]
    DataSource {
        data_source: Arc<str>,
        #[source]
        error: DataSourceError,
    },
}

impl ChutoroError {
    /// Retrieve the stable [`ChutoroErrorCode`] for this error.
    pub const fn code(&self) -> ChutoroErrorCode {
        match self {
            Self::InvalidMinClusterSize { .. } => ChutoroErrorCode::InvalidMinClusterSize,
            Self::EmptySource { .. } => ChutoroErrorCode::EmptySource,
            Self::InsufficientItems { .. } => ChutoroErrorCode::InsufficientItems,
            Self::BackendUnavailable { .. } => ChutoroErrorCode::BackendUnavailable,
            Self::DataSource { .. } => ChutoroErrorCode::DataSourceFailure,
        }
    }

    /// Retrieve the inner [`DataSourceErrorCode`] when the error originated in a [`DataSource`].
    pub const fn data_source_code(&self) -> Option<DataSourceErrorCode> {
        match self {
            Self::DataSource { error, .. } => Some(error.code()),
            _ => None,
        }
    }
}

/// Convenient alias for results returned by the core API.
pub type Result<T> = core::result::Result<T, ChutoroError>;
