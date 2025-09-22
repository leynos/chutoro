//! Error types for the Chutoro core library.
//!
//! Defines error enums exposed by the public API and a convenient result alias.

use std::{num::NonZeroUsize, sync::Arc};

use thiserror::Error;

use crate::builder::ExecutionStrategy;

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

/// Convenient alias for results returned by the core API.
pub type Result<T> = core::result::Result<T, ChutoroError>;
