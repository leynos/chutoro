//! Typed failures emitted by dataset recipes and their I/O ports.

mod cleanup_failure;
mod payloads;

pub use cleanup_failure::CleanupFailure;
pub use payloads::{FetchSizeExceeded, IntegrityFailure, PortFailure};

use std::{error::Error, sync::Arc};

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
    /// A source checksum could not be parsed.
    #[error("integrity failed: {0}")]
    Integrity(#[source] Box<IntegrityFailure>),
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
    /// Staging locks are unavailable on this target.
    #[error("staging locks are not supported on this target")]
    StagingLockUnsupportedPlatform,
    /// The requested staging lockfile name is invalid.
    #[error("invalid staging lock name: {0}")]
    StagingLockInvalidName(Arc<str>),
    /// A staging lock is held by this process or by another holder.
    #[error("staging lock {path} is held; waited {waited_millis} ms")]
    StagingLockContended {
        /// Lockfile path reported to the caller.
        path: Arc<str>,
        /// Milliseconds spent before giving up; zero for in-process rejection.
        waited_millis: u64,
    },
    /// Opening, creating, or locking a staging file failed.
    #[error("staging lock {path} failed: {source}")]
    StagingLockIo {
        /// Lockfile path reported to the caller.
        path: Arc<str>,
        /// Operating-system failure.
        #[source]
        source: std::io::Error,
    },
    /// Cleanup failed after a phase error.
    ///
    /// The driver creates this variant through
    /// [`RecipeError::cleanup_with_phase_error`], which stores a
    /// [`CleanupFailure`] in `source`. That payload retains both the cleanup
    /// failure and the original phase error, and exposes the phase error as
    /// its source chain. [`RecipeError::cleanup`] stores only the cleanup
    /// failure. Use [`RecipeError::cleanup_failure`] to borrow the retained
    /// pair, and [`CleanupFailure::cleanup_error`] to borrow the cleanup
    /// failure from a driver-created payload.
    #[error("cleanup failed in phase {phase:?}: {source}")]
    Cleanup {
        /// Phase that failed before cleanup ran.
        phase: Phase,
        /// Cleanup failure source.
        ///
        /// [`RecipeError::cleanup_with_phase_error`] stores a boxed
        /// [`CleanupFailure`] here so both failures survive. The original
        /// phase error is that payload's source, so [`Error::source`] walks
        /// from the cleanup failure to the reason the recipe failed and on to
        /// any error the phase error wraps.
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

    /// Create an integrity failure from a typed parse failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{IntegrityFailure, RecipeError};
    ///
    /// let error = RecipeError::integrity(IntegrityFailure::MissingAlgorithmPrefix);
    /// assert!(matches!(error, RecipeError::Integrity(_)));
    /// assert_eq!(
    ///     error.to_string(),
    ///     "integrity failed: checksum is missing an algorithm prefix",
    /// );
    /// ```
    #[must_use]
    pub fn integrity(failure: IntegrityFailure) -> Self {
        Self::Integrity(Box::new(failure))
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

    /// Create an unsupported-platform staging-lock failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// assert!(matches!(
    ///     RecipeError::staging_lock_unsupported_platform(),
    ///     RecipeError::StagingLockUnsupportedPlatform
    /// ));
    /// ```
    #[must_use]
    pub const fn staging_lock_unsupported_platform() -> Self {
        Self::StagingLockUnsupportedPlatform
    }

    /// Create an invalid staging-lock name failure from the rejected value.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let error = RecipeError::staging_lock_invalid_name("../escape.lock");
    /// assert!(matches!(error, RecipeError::StagingLockInvalidName(_)));
    /// ```
    #[must_use]
    pub fn staging_lock_invalid_name(value: impl Into<Arc<str>>) -> Self {
        Self::StagingLockInvalidName(value.into())
    }

    /// Create a contended staging-lock failure.
    ///
    /// `waited_millis` is zero when the process-wide registry rejected a
    /// duplicate acquisition before any system call.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let error = RecipeError::staging_lock_contended("cache/data.lock", 0);
    /// assert!(matches!(
    ///     error,
    ///     RecipeError::StagingLockContended { waited_millis: 0, .. }
    /// ));
    /// ```
    #[must_use]
    pub fn staging_lock_contended(path: impl Into<Arc<str>>, waited_millis: u64) -> Self {
        Self::StagingLockContended {
            path: path.into(),
            waited_millis,
        }
    }

    /// Create a staging-lock I/O failure while preserving its source.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::RecipeError;
    ///
    /// let source = std::io::Error::other("open failed");
    /// let error = RecipeError::staging_lock_io("cache/data.lock", source);
    /// assert!(matches!(error, RecipeError::StagingLockIo { .. }));
    /// ```
    #[must_use]
    pub fn staging_lock_io(path: impl Into<Arc<str>>, source: std::io::Error) -> Self {
        Self::StagingLockIo {
            path: path.into(),
            source,
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

const _: () = assert!(std::mem::size_of::<RecipeError>() <= 32);

// `RecipeError` boxes cleanup payloads as `dyn Error + Send + Sync`, so the
// compound failure must satisfy the same bounds as every other boxed source.
const _: fn() = || {
    const fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    assert_send_sync_static::<CleanupFailure>();
};
