//! Error types for the Chutoro core library.
//!
//! Defines error enums exposed by the public API and a convenient result alias.

use std::{fmt, num::NonZeroUsize, sync::Arc};

use thiserror::Error;

use crate::builder::ExecutionStrategy;

macro_rules! define_error_codes {
    (
        $(#[$enum_meta:meta])*
        enum $CodeTy:ident for $ErrTy:ident {
            $(
                $(#[$variant_meta:meta])*
                $CodeVariant:ident => $ErrVariant:ident $( { $($pattern:tt)* } )? => $code:expr
            ),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        #[non_exhaustive]
        pub enum $CodeTy {
            $(
                $(#[$variant_meta])*
                $CodeVariant,
            )+
        }

        impl $CodeTy {
            /// Return the stable machine-readable representation of this error code.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$CodeVariant => $code,)+
                }
            }
        }

        impl fmt::Display for $CodeTy {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl $ErrTy {
            #[doc = concat!(
                "Retrieve the stable [`",
                stringify!($CodeTy),
                "`] for this error."
            )]
            pub const fn code(&self) -> $CodeTy {
                match self {
                    $(Self::$ErrVariant $( { $($pattern)* } )? => $CodeTy::$CodeVariant,)+
                }
            }
        }
    };
}

/// An error produced by [`crate::DataSource`] operations.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum DataSourceError {
    /// Requested index was outside the source's bounds.
    #[error("index {index} is out of bounds")]
    OutOfBounds {
        /// The requested row that exceeded the source bounds.
        index: usize,
    },
    /// Provided output buffer length did not match number of pairs.
    #[error("output buffer has length {out} but {expected} pairs were given")]
    OutputLengthMismatch {
        /// Caller-provided buffer length.
        out: usize,
        /// Expected number of vector pairs required for the operation.
        expected: usize,
    },
    /// Compared vectors had different dimensions.
    #[error("dimension mismatch: left={left}, right={right}")]
    DimensionMismatch {
        /// Dimensionality of the left-hand vector.
        left: usize,
        /// Dimensionality of the right-hand vector.
        right: usize,
    },
    /// Data source contained no rows.
    #[error("data source contains no rows")]
    EmptyData,
    /// Data source rows must have positive dimension.
    #[error("data source vectors must have positive dimension")]
    ZeroDimension,
}

define_error_codes! {
    /// Stable codes describing [`DataSourceError`] variants.
    enum DataSourceErrorCode for DataSourceError {
        /// Requested index was outside the source's bounds.
        OutOfBounds => OutOfBounds { .. } => "DATA_SOURCE_OUT_OF_BOUNDS",
        /// Provided output buffer length did not match number of pairs.
        OutputLengthMismatch => OutputLengthMismatch { .. } => "DATA_SOURCE_OUTPUT_LENGTH_MISMATCH",
        /// Compared vectors had different dimensions.
        DimensionMismatch => DimensionMismatch { .. } => "DATA_SOURCE_DIMENSION_MISMATCH",
        /// Data source contained no rows.
        EmptyData => EmptyData => "DATA_SOURCE_EMPTY",
        /// Data source rows must have positive dimension.
        ZeroDimension => ZeroDimension => "DATA_SOURCE_ZERO_DIMENSION",
    }
}

/// Error type produced when constructing or running [`crate::Chutoro`].
#[non_exhaustive]
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ChutoroError {
    /// Minimum cluster size must be greater than zero.
    #[error("min_cluster_size must be at least 1 (got {got})")]
    InvalidMinClusterSize {
        /// The invalid minimum cluster size supplied by the caller.
        got: usize,
    },
    /// The supplied [`crate::DataSource`] contained no items.
    #[error("data source `{data_source}` contains no items")]
    EmptySource {
        /// Identifier for the empty data source.
        data_source: Arc<str>,
    },
    /// The [`crate::DataSource`] did not contain enough items for the configured
    /// `min_cluster_size`.
    #[error(
        "data source `{data_source}` has {items} items but min_cluster_size requires {min_cluster_size}"
    )]
    InsufficientItems {
        /// Identifier for the data source that lacked sufficient items.
        data_source: Arc<str>,
        /// Number of items available in the data source.
        items: usize,
        /// Minimum cluster size required by the algorithm.
        min_cluster_size: NonZeroUsize,
    },
    /// The requested execution strategy is unavailable in the current build.
    #[error("the requested execution strategy {requested:?} is not available in this build")]
    BackendUnavailable {
        /// Strategy that could not be satisfied by the current build.
        requested: ExecutionStrategy,
    },
    /// A [`crate::DataSource`] operation failed while running the algorithm.
    #[error("data source `{data_source}` failed: {error}")]
    DataSource {
        /// Identifier for the data source that produced the error.
        data_source: Arc<str>,
        #[source]
        /// Underlying data source error bubbled up by the algorithm.
        error: DataSourceError,
    },
}

define_error_codes! {
    /// Stable codes describing [`ChutoroError`] variants.
    enum ChutoroErrorCode for ChutoroError {
        /// Minimum cluster size must be greater than zero.
        InvalidMinClusterSize => InvalidMinClusterSize { .. } => "CHUTORO_INVALID_MIN_CLUSTER_SIZE",
        /// The supplied [`crate::DataSource`] contained no items.
        EmptySource => EmptySource { .. } => "CHUTORO_EMPTY_SOURCE",
        /// The [`crate::DataSource`] did not contain enough items for the configured
        /// minimum cluster size.
        InsufficientItems => InsufficientItems { .. } => "CHUTORO_INSUFFICIENT_ITEMS",
        /// The requested execution strategy is unavailable in the current build.
        BackendUnavailable => BackendUnavailable { .. } => "CHUTORO_BACKEND_UNAVAILABLE",
        /// A [`crate::DataSource`] operation failed while running the algorithm.
        DataSourceFailure => DataSource { .. } => "CHUTORO_DATA_SOURCE_FAILURE",
    }
}

impl ChutoroError {
    /// Retrieve the inner [`DataSourceErrorCode`] when the error originated in a [`crate::DataSource`].
    pub const fn data_source_code(&self) -> Option<DataSourceErrorCode> {
        match self {
            Self::DataSource { error, .. } => Some(error.code()),
            _ => None,
        }
    }
}

/// Convenient alias for results returned by the core API.
pub type Result<T> = core::result::Result<T, ChutoroError>;
