//! Pure, clock-free policy for bounded staging-lock acquisition.
//!
//! The adapter owns the clock and the filesystem. This module owns only the
//! decision: given the outcome of one non-blocking attempt, how long the caller
//! has already waited, and whether a contention warning has been emitted, it
//! says whether to keep the lock, retry after a bounded sleep, or give up.
//!
//! Keeping the decision pure makes the hard-timeout rules, the sleep bound,
//! and the exactly-one-warning rule testable over arbitrary traces without a
//! real clock or a real lock.

use std::time::Duration;

use thiserror::Error;

/// Default interval between non-blocking lock attempts.
///
/// The interval is short enough to acquire a lock promptly after its holder
/// releases it, and long enough that a contended acquisition does not spin.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Smallest accepted poll interval.
///
/// A zero interval would turn the bounded retry loop into a busy loop, so a
/// policy rejects anything faster than one attempt per millisecond.
pub const MIN_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Invalid [`LockPolicy`] construction.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum LockPolicyError {
    /// The acquisition timeout was zero.
    #[error("staging lock timeout must be greater than zero")]
    ZeroTimeout,
    /// The poll interval was below [`MIN_POLL_INTERVAL`].
    #[error("staging lock poll interval must be at least {minimum:?}")]
    PollIntervalTooShort {
        /// Smallest accepted poll interval.
        minimum: Duration,
    },
}

/// Result of one non-blocking exclusive `flock` attempt.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LockAttempt {
    /// The kernel granted the exclusive lock.
    Acquired,
    /// Another open file description holds the lock (`WOULDBLOCK`).
    Contended,
    /// A signal interrupted the attempt (`INTR`).
    Interrupted,
}

/// Decision produced by [`LockPolicy::next_step`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LockStep {
    /// Keep the granted lock; the attempt succeeded before the deadline.
    Acquired,
    /// Sleep for `delay`, then attempt again.
    Retry {
        /// Sleep before the next attempt, never longer than the time left
        /// before the deadline.
        delay: Duration,
        /// True when the caller should emit its first contention warning.
        warn: bool,
    },
    /// The deadline expired, or a late success must be released.
    TimedOut,
}

/// Bounded acquisition policy for one staging lock.
///
/// A policy requires a non-zero timeout and accepts a poll interval of at
/// least [`MIN_POLL_INTERVAL`]. The default poll interval is
/// [`DEFAULT_POLL_INTERVAL`].
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use chutoro_bench_datasets::{LockAttempt, LockPolicy, LockStep};
///
/// let policy = LockPolicy::new(Duration::from_millis(100))?;
/// assert_eq!(policy.poll_interval(), Duration::from_millis(25));
///
/// let step = policy.next_step(
///     LockAttempt::Contended,
///     Duration::from_millis(10),
///     false,
/// );
/// assert!(matches!(step, LockStep::Retry { warn: true, .. }));
/// # Ok::<(), chutoro_bench_datasets::LockPolicyError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LockPolicy {
    /// Hard deadline measured from the first attempt.
    timeout: Duration,
    /// Maximum sleep between attempts.
    poll_interval: Duration,
}

impl LockPolicy {
    /// Create a policy with a hard timeout and the default poll interval.
    ///
    /// # Errors
    ///
    /// Returns [`LockPolicyError::ZeroTimeout`] when `timeout` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use chutoro_bench_datasets::{LockPolicy, LockPolicyError};
    ///
    /// assert!(LockPolicy::new(Duration::from_millis(1)).is_ok());
    /// assert_eq!(
    ///     LockPolicy::new(Duration::ZERO),
    ///     Err(LockPolicyError::ZeroTimeout),
    /// );
    /// ```
    pub const fn new(timeout: Duration) -> Result<Self, LockPolicyError> {
        if timeout.is_zero() {
            return Err(LockPolicyError::ZeroTimeout);
        }
        Ok(Self {
            timeout,
            poll_interval: DEFAULT_POLL_INTERVAL,
        })
    }

    /// Replace the poll interval.
    ///
    /// # Errors
    ///
    /// Returns [`LockPolicyError::PollIntervalTooShort`] when `poll_interval`
    /// is below [`MIN_POLL_INTERVAL`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use chutoro_bench_datasets::{LockPolicy, LockPolicyError};
    ///
    /// let policy = LockPolicy::new(Duration::from_secs(1))?
    ///     .with_poll_interval(Duration::from_millis(5))?;
    /// assert_eq!(policy.poll_interval(), Duration::from_millis(5));
    ///
    /// assert_eq!(
    ///     policy.with_poll_interval(Duration::ZERO),
    ///     Err(LockPolicyError::PollIntervalTooShort {
    ///         minimum: Duration::from_millis(1),
    ///     }),
    /// );
    /// # Ok::<(), LockPolicyError>(())
    /// ```
    pub fn with_poll_interval(mut self, poll_interval: Duration) -> Result<Self, LockPolicyError> {
        if poll_interval < MIN_POLL_INTERVAL {
            return Err(LockPolicyError::PollIntervalTooShort {
                minimum: MIN_POLL_INTERVAL,
            });
        }
        self.poll_interval = poll_interval;
        Ok(self)
    }

    /// Return the hard timeout measured from the first attempt.
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Return the maximum sleep between attempts.
    #[must_use]
    pub const fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Decide what the acquisition loop should do after one attempt.
    ///
    /// A success at or after the deadline is a late result and is reported as
    /// [`LockStep::TimedOut`]; the caller must release the lock it just
    /// received. `warned` records whether the caller has already emitted its
    /// contention warning, so `warn` is true at most once per acquisition.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    /// use chutoro_bench_datasets::{LockAttempt, LockPolicy, LockStep};
    ///
    /// let policy = LockPolicy::new(Duration::from_millis(50))?;
    /// assert_eq!(
    ///     policy.next_step(LockAttempt::Acquired, Duration::from_millis(10), false),
    ///     LockStep::Acquired,
    /// );
    /// assert_eq!(
    ///     policy.next_step(LockAttempt::Contended, Duration::from_millis(50), false),
    ///     LockStep::TimedOut,
    /// );
    /// # Ok::<(), chutoro_bench_datasets::LockPolicyError>(())
    /// ```
    #[must_use]
    pub fn next_step(&self, attempt: LockAttempt, elapsed: Duration, warned: bool) -> LockStep {
        match attempt {
            LockAttempt::Acquired if elapsed < self.timeout => LockStep::Acquired,
            LockAttempt::Acquired => LockStep::TimedOut,
            LockAttempt::Contended => self.retry_step(elapsed, !warned),
            LockAttempt::Interrupted => self.retry_step(elapsed, false),
        }
    }

    /// Build a retry step bounded by the remaining time before the deadline.
    fn retry_step(&self, elapsed: Duration, warn: bool) -> LockStep {
        self.timeout
            .checked_sub(elapsed)
            .filter(|remaining| !remaining.is_zero())
            .map_or(LockStep::TimedOut, |remaining| LockStep::Retry {
                delay: self.poll_interval.min(remaining),
                warn,
            })
    }
}
