//! Error types for synthetic benchmark data generation.

use std::path::PathBuf;

/// Errors that may occur while preparing benchmark data sources.
#[derive(Debug, thiserror::Error)]
pub enum SyntheticError {
    /// The requested point count was zero.
    #[error("point count must be greater than zero")]
    ZeroPoints,
    /// The requested dimension count was zero.
    #[error("dimension count must be greater than zero")]
    ZeroDimensions,
    /// The requested cluster count was zero.
    #[error("cluster count must be greater than zero")]
    ZeroClusters,
    /// The requested manifold turn count was zero.
    #[error("manifold turns must be greater than zero")]
    ZeroTurns,
    /// The requested text alphabet was empty.
    #[error("text alphabet must not be empty")]
    EmptyAlphabet,
    /// The requested text corpus size was zero.
    #[error("text item count must be greater than zero")]
    ZeroTextItems,
    /// The requested minimum text length was zero.
    #[error("minimum text length must be greater than zero")]
    ZeroTextLength,
    /// The requested text length range was invalid.
    #[error("invalid text length range: min={min_length}, max={max_length}")]
    InvalidTextLengthRange {
        /// Minimum configured text length.
        min_length: usize,
        /// Maximum configured text length.
        max_length: usize,
    },
    /// The configured cluster count exceeded the available points.
    #[error("cluster count ({cluster_count}) must not exceed point count ({point_count})")]
    ClusterCountExceedsPointCount {
        /// Number of clusters requested.
        cluster_count: usize,
        /// Number of points requested.
        point_count: usize,
    },
    /// The requested `point_count * dimensions` overflowed `usize`.
    #[error("point_count * dimensions overflows usize")]
    Overflow,
    /// A floating-point generator parameter was invalid.
    #[error("invalid floating-point parameter `{parameter}`")]
    InvalidFloatParameter {
        /// Name of the invalid parameter.
        parameter: &'static str,
    },
    /// An anisotropy axis scale was non-positive or non-finite.
    #[error("anisotropy axis {index} must be finite and greater than zero")]
    InvalidAxisScale {
        /// Zero-based axis index.
        index: usize,
    },
    /// Axis-aligned anisotropy scale count did not match dimensions.
    #[error("anisotropy axis scale length mismatch: expected {expected}, got {actual}")]
    AxisScaleLengthMismatch {
        /// Expected number of axis scales.
        expected: usize,
        /// Received number of axis scales.
        actual: usize,
    },
    /// The selected manifold pattern requires more dimensions.
    #[error("manifold pattern `{pattern}` requires at least {minimum} dimensions (got {actual})")]
    InsufficientManifoldDimensions {
        /// Pattern name.
        pattern: &'static str,
        /// Minimum required dimensions.
        minimum: usize,
        /// Actual configured dimensions.
        actual: usize,
    },
    /// Reading or writing cached dataset files failed.
    #[error("I/O failure while handling cached dataset data: {0}")]
    Io(#[from] std::io::Error),
    /// A dataset download failed.
    #[error("dataset download failed for `{url}`: {message}")]
    Download {
        /// URL that failed.
        url: String,
        /// Human-readable failure message.
        message: String,
    },
    /// A downloaded dataset file was malformed.
    #[error("invalid MNIST file `{path}`: {message}")]
    InvalidMnistFile {
        /// Path of the malformed file.
        path: PathBuf,
        /// Human-readable validation failure.
        message: String,
    },
    /// Two MNIST image files had mismatched dimensions.
    #[error("MNIST image dimensions mismatch between train and test: train={train}, test={test}")]
    MnistDimensionMismatch {
        /// Flattened train image dimensions.
        train: usize,
        /// Flattened test image dimensions.
        test: usize,
    },
}
