//! Nightly gating logic for slow CI jobs such as `make kani-full`.

use std::fmt;

const SECONDS_PER_DAY: u64 = 86_400;
const ALLOWED_FUTURE_SKEW_SECONDS: u64 = 300;

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
/// The comparison uses a rolling 24-hour window derived from Unix epoch
/// seconds. Small future skews (up to 300 seconds) are treated as a skip
/// instead of a hard failure. Set `force` to `true` to bypass the date gate
/// (for manual verification runs).
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
        let skew = commit_epoch - now_epoch;
        if skew <= ALLOWED_FUTURE_SKEW_SECONDS {
            return Ok(NightlyDecision::new(
                false,
                format!("commit timestamp {commit_epoch} is {skew}s ahead of now; skipping"),
            ));
        }

        return Err(NightlyGateError::CommitFromFuture {
            commit_epoch,
            now_epoch,
        });
    }

    if is_within_last_day(commit_epoch, now_epoch) {
        Ok(NightlyDecision::new(
            true,
            "main updated within last 24 hours (UTC); running Kani",
        ))
    } else {
        Ok(NightlyDecision::new(
            false,
            "no main commits in last 24 hours (UTC); skipping",
        ))
    }
}

fn is_within_last_day(commit_epoch: u64, now_epoch: u64) -> bool {
    commit_epoch >= now_epoch.saturating_sub(SECONDS_PER_DAY)
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::rstest;

    const DAY: u64 = SECONDS_PER_DAY;

    #[rstest]
    #[case::same_timestamp(DAY, DAY, true)]
    #[case::within_window(DAY + 10, DAY + 100, true)]
    #[case::boundary_window(DAY, DAY * 2, true)]
    #[case::outside_window(DAY - 1, DAY * 2, false)]
    fn run_decision_by_window(
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
    #[case::small_future_skew(DAY + 10, DAY + 9)]
    #[case::boundary_future_skew(DAY + ALLOWED_FUTURE_SKEW_SECONDS, DAY)]
    fn small_future_skew_is_skip(#[case] commit_epoch: u64, #[case] now_epoch: u64) {
        let decision = match should_run_kani_full(commit_epoch, now_epoch, false) {
            Ok(decision) => decision,
            Err(error) => panic!("expected small skew to skip, got {error}"),
        };

        assert!(!decision.should_run);
        assert!(decision.reason.contains("ahead"));
    }

    #[rstest]
    #[case::recent_commit(DAY * 3 - 1, DAY * 3, true)]
    #[case::stale_commit(DAY * 2 - 1, DAY * 3, false)]
    fn window_boundary_cases(
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
