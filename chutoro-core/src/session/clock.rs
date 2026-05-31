//! Monotonic clock seam for [`super::ClusteringSession`].
//!
//! Provides an injectable abstraction over [`std::time::Instant`] so that
//! per-point insertion latency recorded by [`super::ClusteringSession::append`]
//! is deterministic in tests. Only compiled when the `metrics` Cargo feature
//! is enabled.

use std::time::Instant;

/// Abstracts monotonic time measurement for dependency injection.
///
/// Implement this trait to replace wall-clock reads in
/// [`super::ClusteringSession::append`] with a controllable time source.
pub(crate) trait MonotonicClock: Send + Sync + std::fmt::Debug {
    /// Returns the current instant.
    fn now(&self) -> Instant;
}

/// Production implementation backed by [`std::time::Instant::now`].
#[derive(Debug, Default)]
pub(super) struct StdMonotonicClock;

impl MonotonicClock for StdMonotonicClock {
    #[inline]
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Test double: returns pre-seeded [`Instant`]s in sequence.
///
/// Construct with [`FixedMonotonicClock::with_elapsed`]; calls to [`now`]
/// return `start` on the first call and `start + elapsed` on the second.
/// Subsequent calls panic to detect incorrect test setup.
#[cfg(test)]
#[derive(Debug)]
pub(super) struct FixedMonotonicClock {
    instants: std::sync::Mutex<std::collections::VecDeque<Instant>>,
}

#[cfg(test)]
impl FixedMonotonicClock {
    /// Constructs a clock that returns `start` then `start + elapsed` on
    /// successive calls to [`now`].
    pub(super) fn with_elapsed(elapsed: std::time::Duration) -> Self {
        let start = Instant::now();
        Self {
            instants: std::sync::Mutex::new([start, start + elapsed].into_iter().collect()),
        }
    }
}

#[cfg(test)]
impl MonotonicClock for FixedMonotonicClock {
    fn now(&self) -> Instant {
        self.instants
            .lock()
            .expect("FixedMonotonicClock mutex poisoned")
            .pop_front()
            .expect("FixedMonotonicClock: now() called more times than seeded instants")
    }
}
