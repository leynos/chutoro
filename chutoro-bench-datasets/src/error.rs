//! Typed failures emitted by dataset recipes and their I/O ports.

use std::{error::Error, fmt, sync::Arc};

use thiserror::Error;

use crate::{Phase, PortName, SourceUrl};

/// Failure raised by a dataset recipe, driver, or port adapter.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum RecipeError {
    /// The supplied source string uses an unsupported URL scheme.
    #[error("invalid source URL: {0}")]
    InvalidSource(Arc<str>),
    /// The supplied recipe version is not `major.minor.patch`.
    #[error("invalid recipe version: {0}")]
    InvalidVersion(Arc<str>),
    /// Checksum support is deferred to roadmap item `10.1.2`.
    #[error("checksum schemes are not supported until roadmap 10.1.2")]
    ChecksumUnsupported,
    /// A port failed while servicing the recipe.
    #[error("{0}")]
    Port(PortFailure),
    /// Validation rejected fetched data.
    #[error("validate failed: {0}")]
    Validate(Arc<str>),
    /// Preparation failed after validation.
    #[error("prepare failed: {0}")]
    Prepare(Arc<str>),
    /// Publishing failed after preparation.
    #[error("publish failed: {0}")]
    Publish(Arc<str>),
    /// Fetching would exceed the mandatory byte cap.
    #[error("{0}")]
    FetchSizeExceeded(FetchSizeExceeded),
    /// Cleanup failed after a phase error.
    #[error("cleanup failed in phase {phase:?}: {source}")]
    Cleanup {
        /// Phase that failed before cleanup ran.
        phase: Phase,
        /// Cleanup failure source.
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
    /// Opaque adapter-specific failure.
    #[error(transparent)]
    Other(Box<dyn Error + Send + Sync>),
}

impl RecipeError {
    /// Create an invalid-source failure from the rejected value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let error = RecipeError::invalid_source("ftp://example.test/data.bin");
    /// assert!(matches!(error, RecipeError::InvalidSource(_)));
    /// ```
    #[must_use]
    pub fn invalid_source(value: impl Into<Arc<str>>) -> Self {
        Self::InvalidSource(value.into())
    }

    /// Create an invalid-version failure from the rejected value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let error = RecipeError::invalid_version("1.2");
    /// assert!(matches!(error, RecipeError::InvalidVersion(_)));
    /// ```
    #[must_use]
    pub fn invalid_version(value: impl Into<Arc<str>>) -> Self {
        Self::InvalidVersion(value.into())
    }

    /// Create a port failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{PortName, RecipeError};
    ///
    /// let error = RecipeError::port(PortName::Fetcher, "source missing");
    /// assert!(matches!(error, RecipeError::Port(_)));
    /// assert_eq!(error.to_string(), "port Fetcher failed: source missing");
    /// ```
    #[must_use]
    pub fn port(port: PortName, reason: impl Into<Arc<str>>) -> Self {
        Self::Port(PortFailure {
            port,
            reason: reason.into(),
        })
    }

    /// Create a validation failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let error = RecipeError::validate("manifest missing");
    /// assert!(matches!(error, RecipeError::Validate(_)));
    /// assert_eq!(error.to_string(), "validate failed: manifest missing");
    /// ```
    #[must_use]
    pub fn validate(reason: impl Into<Arc<str>>) -> Self {
        Self::Validate(reason.into())
    }

    /// Create a preparation failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let error = RecipeError::prepare("archive corrupt");
    /// assert!(matches!(error, RecipeError::Prepare(_)));
    /// assert_eq!(error.to_string(), "prepare failed: archive corrupt");
    /// ```
    #[must_use]
    pub fn prepare(reason: impl Into<Arc<str>>) -> Self {
        Self::Prepare(reason.into())
    }

    /// Create a publish failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let error = RecipeError::publish("object already exists");
    /// assert!(matches!(error, RecipeError::Publish(_)));
    /// assert_eq!(error.to_string(), "publish failed: object already exists");
    /// ```
    #[must_use]
    pub fn publish(reason: impl Into<Arc<str>>) -> Self {
        Self::Publish(reason.into())
    }

    /// Create a fetch-size failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{FetchSizeExceeded, RecipeError, SourceUrl};
    ///
    /// let source = SourceUrl::parse("https://example.test/data.bin")?;
    /// let error = RecipeError::fetch_size_exceeded(source, 1024);
    /// assert!(matches!(
    ///     error,
    ///     RecipeError::FetchSizeExceeded(FetchSizeExceeded {
    ///         limit_bytes: 1024,
    ///         ..
    ///     })
    /// ));
    /// # Ok::<(), RecipeError>(())
    /// ```
    #[must_use]
    pub const fn fetch_size_exceeded(url: SourceUrl, limit_bytes: usize) -> Self {
        Self::FetchSizeExceeded(FetchSizeExceeded { url, limit_bytes })
    }

    /// Create a cleanup failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Phase, RecipeError};
    ///
    /// let source = std::io::Error::other("remove cache");
    /// let error = RecipeError::cleanup(Phase::Prepare, source);
    /// assert!(matches!(error, RecipeError::Cleanup { phase: Phase::Prepare, .. }));
    /// ```
    #[must_use]
    pub fn cleanup(phase: Phase, source: impl Error + Send + Sync + 'static) -> Self {
        Self::Cleanup {
            phase,
            source: Box::new(source),
        }
    }

    /// Create an opaque adapter-specific failure while preserving its source.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let source = std::io::Error::other("adapter failed");
    /// let error = RecipeError::other(source);
    /// assert!(matches!(error, RecipeError::Other(_)));
    /// assert_eq!(error.to_string(), "adapter failed");
    /// ```
    #[must_use]
    pub fn other(source: impl Error + Send + Sync + 'static) -> Self {
        Self::Other(Box::new(source))
    }
}

/// Port failure payload useful for tests and future adapters.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortFailure {
    /// Port that failed.
    pub port: PortName,
    /// Human-readable reason.
    pub reason: Arc<str>,
}

impl fmt::Display for PortFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "port {:?} failed: {}", self.port, self.reason)
    }
}

/// Fetch-size failure payload useful for tests and future adapters.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FetchSizeExceeded {
    /// Source that exceeded the cap.
    pub url: SourceUrl,
    /// Maximum byte count allowed.
    pub limit_bytes: usize,
}

impl fmt::Display for FetchSizeExceeded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "fetch exceeded max_bytes={}: {}",
            self.limit_bytes, self.url
        )
    }
}

const _: () = assert!(std::mem::size_of::<RecipeError>() <= 32);
