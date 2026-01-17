//! Nightly gating logic for slow CI jobs such as `make kani-full`.

use std::fmt;

const SECONDS_PER_DAY: u64 = 86_400;

/// Decision describing whether the nightly Kani job should run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NightlyDecision {
    /// `true` when the job should run.
    pub should_run: bool,
    /// Human-readable reason for the decision.
    pub reason: String,
}

impl NightlyDecision {
    fn new(should_run: bool, reason: impl Into<String>) -> Self {
        Self {
            should_run,
            reason: reason.into(),
        }
    }
}

/// Errors surfaced when nightly gating encounters invalid timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NightlyGateError {
    /// The commit timestamp is later than the current time.
    CommitFromFuture {
        /// The commit epoch (seconds since Unix epoch).
        commit_epoch: u64,
        /// The current epoch (seconds since Unix epoch).
        now_epoch: u64,
    },
}

impl fmt::Display for NightlyGateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommitFromFuture {
                commit_epoch,
                now_epoch,
            } => write!(
                formatter,
                "commit timestamp {commit_epoch} is after current time {now_epoch}"
            ),
        }
    }
}

impl std::error::Error for NightlyGateError {}

/// Decide whether the nightly Kani job should run.
///
/// The comparison uses UTC day boundaries derived from Unix epoch seconds. Set
/// `force` to `true` to bypass the date gate (for manual verification runs).
///
/// # Examples
///
/// ```
/// use chutoro_test_support::ci::nightly_gate::should_run_kani_full;
///
/// let decision = should_run_kani_full(86_400, 86_401, false)
///     .expect("valid timestamps");
/// assert!(decision.should_run);
/// assert!(decision.reason.contains("UTC"));
/// ```
pub fn should_run_kani_full(
    commit_epoch: u64,
    now_epoch: u64,
    force: bool,
) -> Result<NightlyDecision, NightlyGateError> {
    if force {
        return Ok(NightlyDecision::new(true, "forced run requested"));
    }

    if commit_epoch > now_epoch {
        return Err(NightlyGateError::CommitFromFuture {
            commit_epoch,
            now_epoch,
        });
    }

    let commit_day = utc_day(commit_epoch);
    let now_day = utc_day(now_epoch);

    if commit_day == now_day {
        Ok(NightlyDecision::new(
            true,
            "main updated today (UTC); running Kani",
        ))
    } else {
        Ok(NightlyDecision::new(
            false,
            "no main commits today (UTC); skipping",
        ))
    }
}

fn utc_day(epoch_seconds: u64) -> u64 {
    epoch_seconds / SECONDS_PER_DAY
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;

    const DAY: u64 = SECONDS_PER_DAY;

    #[rstest]
    #[case::same_day(DAY + 1, DAY + 10, true)]
    #[case::same_day_later(DAY * 2 + 5, DAY * 2 + 50, true)]
    #[case::previous_day(DAY - 1, DAY + 1, false)]
    #[case::two_days_ago(DAY, DAY * 3, false)]
    fn run_decision_by_day(
        #[case] commit_epoch: u64,
        #[case] now_epoch: u64,
        #[case] expected_run: bool,
    ) {
        let decision = match should_run_kani_full(commit_epoch, now_epoch, false) {
            Ok(decision) => decision,
            Err(error) => panic!("expected valid timestamps, got {error}"),
        };

        assert_eq!(decision.should_run, expected_run);
    }

    #[rstest]
    #[case::force_same_day(DAY + 1, DAY + 2)]
    #[case::force_previous_day(DAY - 1, DAY + 2)]
    #[case::force_future(DAY * 3, DAY + 2)]
    fn force_override_always_runs(#[case] commit_epoch: u64, #[case] now_epoch: u64) {
        let decision = match should_run_kani_full(commit_epoch, now_epoch, true) {
            Ok(decision) => decision,
            Err(error) => panic!("force override should not error: {error}"),
        };

        assert!(decision.should_run);
        assert!(decision.reason.contains("forced"));
    }

    #[rstest]
    #[case::future_commit(DAY * 2, DAY + 1)]
    #[case::far_future_commit(DAY * 10, DAY * 3)]
    fn future_commit_is_error(#[case] commit_epoch: u64, #[case] now_epoch: u64) {
        match should_run_kani_full(commit_epoch, now_epoch, false) {
            Ok(_) => panic!("future commit should fail"),
            Err(error) => {
                assert_eq!(
                    error,
                    NightlyGateError::CommitFromFuture {
                        commit_epoch,
                        now_epoch,
                    }
                );
            }
        }
    }

    #[rstest]
    #[case::before_midnight(DAY - 1, DAY, false)]
    #[case::after_midnight(DAY, DAY + 1, true)]
    fn midnight_boundary_cases(
        #[case] commit_epoch: u64,
        #[case] now_epoch: u64,
        #[case] expected_run: bool,
    ) {
        let decision = match should_run_kani_full(commit_epoch, now_epoch, false) {
            Ok(decision) => decision,
            Err(error) => panic!("expected valid timestamps, got {error}"),
        };

        assert_eq!(decision.should_run, expected_run);
    }
}
