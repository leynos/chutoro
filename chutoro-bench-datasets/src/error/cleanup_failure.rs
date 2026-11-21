//! Payload retaining both failures when a recipe phase and its cleanup fail.

use thiserror::Error;

use super::RecipeError;

/// Both failures retained when a recipe phase fails and cleanup also fails.
///
/// [`RecipeError::cleanup_with_phase_error`] stores this payload inside
/// [`RecipeError::Cleanup`]. The cleanup error is the outer failure and the
/// original phase error is its source, so
/// [`std::error::Error::source`] walks from the cleanup failure to the reason
/// the recipe failed and on to any error the phase error wraps.
#[derive(Debug, Error)]
#[error("{cleanup_error}; original phase error: {phase_error}")]
pub struct CleanupFailure {
    /// Error returned by the phase that failed before cleanup ran.
    #[source]
    phase_error: RecipeError,
    /// Error returned by the recipe cleanup hook.
    cleanup_error: RecipeError,
}

impl CleanupFailure {
    /// Bundle the phase error and the cleanup error.
    pub(super) const fn new(phase_error: RecipeError, cleanup_error: RecipeError) -> Self {
        Self {
            phase_error,
            cleanup_error,
        }
    }

    /// Borrow the error returned by the phase that failed before cleanup.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Phase, PortName, RecipeError};
    ///
    /// let error = RecipeError::cleanup_with_phase_error(
    ///     Phase::Prepare,
    ///     RecipeError::prepare("archive corrupt"),
    ///     RecipeError::port(PortName::Storage, "cache locked"),
    /// );
    /// let cleanup = error
    ///     .cleanup_failure()
    ///     .ok_or_else(|| std::io::Error::other("cleanup failure should be retained"))?;
    /// assert_eq!(
    ///     cleanup.phase_error().to_string(),
    ///     "prepare failed: archive corrupt"
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn phase_error(&self) -> &RecipeError {
        &self.phase_error
    }

    /// Borrow the error returned by the recipe cleanup hook.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Phase, PortName, RecipeError};
    ///
    /// let error = RecipeError::cleanup_with_phase_error(
    ///     Phase::Prepare,
    ///     RecipeError::prepare("archive corrupt"),
    ///     RecipeError::port(PortName::Storage, "cache locked"),
    /// );
    /// let cleanup = error
    ///     .cleanup_failure()
    ///     .ok_or_else(|| std::io::Error::other("cleanup failure should be retained"))?;
    /// assert_eq!(
    ///     cleanup.cleanup_error().to_string(),
    ///     "port Storage failed: cache locked"
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub const fn cleanup_error(&self) -> &RecipeError {
        &self.cleanup_error
    }
}
