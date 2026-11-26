//! Payload retaining both failures when a recipe phase and its cleanup fail.

use std::error::Error;

use thiserror::Error;

use super::RecipeError;
use crate::Phase;

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
    /// # Returns
    /// The [`RecipeError`] the failed phase produced before cleanup ran.
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
    /// # Returns
    /// The [`RecipeError`] the cleanup hook produced.
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

impl RecipeError {
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

    /// Create a cleanup failure that retains the original phase error.
    ///
    /// # Parameters
    /// - `phase`: Phase whose execution returned `phase_error`.
    /// - `phase_error`: Error returned by the failed phase. It becomes the
    ///   source of the retained cleanup failure.
    /// - `cleanup_error`: Error returned by the recipe cleanup hook.
    ///
    /// # Returns
    /// [`RecipeError::Cleanup`] whose source is a boxed [`CleanupFailure`]
    /// retaining both errors.
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
    /// assert!(matches!(
    ///     error,
    ///     RecipeError::Cleanup {
    ///         phase: Phase::Prepare,
    ///         ..
    ///     }
    /// ));
    /// ```
    #[must_use]
    pub fn cleanup_with_phase_error(phase: Phase, phase_error: Self, cleanup_error: Self) -> Self {
        Self::Cleanup {
            phase,
            source: Box::new(CleanupFailure::new(phase_error, cleanup_error)),
        }
    }

    /// Borrow the failures retained by a driver-created cleanup error.
    ///
    /// # Returns
    /// [`Some`] when the `Cleanup` source is a [`CleanupFailure`], which is
    /// what [`RecipeError::cleanup_with_phase_error`] stores. Returns [`None`]
    /// for every other variant and for a `Cleanup` whose source is any other
    /// error, including one built through [`RecipeError::cleanup`] with an
    /// ordinary cleanup error.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_bench_datasets::{Phase, PortName, RecipeError};
    ///
    /// let error = RecipeError::cleanup_with_phase_error(
    ///     Phase::Fetch,
    ///     RecipeError::port(PortName::Fetcher, "source missing"),
    ///     RecipeError::port(PortName::Storage, "cache locked"),
    /// );
    /// assert!(error.cleanup_failure().is_some());
    /// assert!(RecipeError::validate("bad manifest").cleanup_failure().is_none());
    /// ```
    #[must_use]
    pub fn cleanup_failure(&self) -> Option<&CleanupFailure> {
        match self {
            Self::Cleanup { source, .. } => source.downcast_ref::<CleanupFailure>(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the downcast that finds a retained cleanup payload.

    use super::CleanupFailure;
    use crate::{Phase, PortName, RecipeError};

    /// `cleanup_failure()` reports the payload type rather than which
    /// constructor stored it, so the two-argument constructor can also carry a
    /// compound payload and still expose it.
    #[test]
    fn cleanup_failure_is_found_whatever_constructor_stored_it() {
        let payload = CleanupFailure::new(
            RecipeError::prepare("archive corrupt"),
            RecipeError::port(PortName::Storage, "cache locked"),
        );
        let rebuilt = RecipeError::cleanup(Phase::Prepare, payload);

        let found = rebuilt
            .cleanup_failure()
            .expect("a stored CleanupFailure payload should stay reachable");
        assert_eq!(
            found.cleanup_error().to_string(),
            "port Storage failed: cache locked"
        );
        assert_eq!(
            found.phase_error().to_string(),
            "prepare failed: archive corrupt"
        );
    }
}
