//! Unit and property tests for the staging-lock acquisition policy.

#![cfg(feature = "http")]

use std::time::Duration;

use chutoro_bench_datasets::{
    DEFAULT_POLL_INTERVAL, LockAttempt, LockPolicy, LockPolicyError, LockStep, MIN_POLL_INTERVAL,
};
use proptest::prelude::*;
use rstest::rstest;

/// Build the policy used by the table tests: 100 ms timeout, 10 ms poll.
fn policy() -> Result<LockPolicy, LockPolicyError> {
    LockPolicy::new(Duration::from_millis(100))
        .and_then(|policy| policy.with_poll_interval(Duration::from_millis(10)))
}

/// One row of the `next_step` truth table: inputs and the exact decision.
struct StepCase {
    /// Attempt outcome presented to the policy.
    attempt: LockAttempt,
    /// Time already spent when the attempt completed.
    elapsed_ms: u64,
    /// Whether a contention warning has already been emitted.
    warned: bool,
    /// The decision the policy must return.
    expected: LockStep,
}

#[rstest]
#[case::acquired_before_deadline(StepCase {
    attempt: LockAttempt::Acquired,
    elapsed_ms: 99,
    warned: false,
    expected: LockStep::Acquired,
})]
#[case::acquired_at_deadline(StepCase {
    attempt: LockAttempt::Acquired,
    elapsed_ms: 100,
    warned: false,
    expected: LockStep::TimedOut { warn: false },
})]
#[case::acquired_after_deadline(StepCase {
    attempt: LockAttempt::Acquired,
    elapsed_ms: 101,
    warned: false,
    expected: LockStep::TimedOut { warn: false },
})]
#[case::contended_at_deadline_warns(StepCase {
    attempt: LockAttempt::Contended,
    elapsed_ms: 100,
    warned: false,
    expected: LockStep::TimedOut { warn: true },
})]
#[case::contended_after_deadline_warns(StepCase {
    attempt: LockAttempt::Contended,
    elapsed_ms: 101,
    warned: false,
    expected: LockStep::TimedOut { warn: true },
})]
#[case::contended_at_deadline_after_warning(StepCase {
    attempt: LockAttempt::Contended,
    elapsed_ms: 100,
    warned: true,
    expected: LockStep::TimedOut { warn: false },
})]
#[case::interrupted_at_deadline(StepCase {
    attempt: LockAttempt::Interrupted,
    elapsed_ms: 100,
    warned: false,
    expected: LockStep::TimedOut { warn: false },
})]
#[case::contended_uses_poll(StepCase {
    attempt: LockAttempt::Contended,
    elapsed_ms: 10,
    warned: false,
    expected: LockStep::Retry {
        delay: Duration::from_millis(10),
        warn: true,
    },
})]
#[case::contended_clamps_to_remaining(StepCase {
    attempt: LockAttempt::Contended,
    elapsed_ms: 95,
    warned: false,
    expected: LockStep::Retry {
        delay: Duration::from_millis(5),
        warn: true,
    },
})]
#[case::contended_does_not_rewarn(StepCase {
    attempt: LockAttempt::Contended,
    elapsed_ms: 10,
    warned: true,
    expected: LockStep::Retry {
        delay: Duration::from_millis(10),
        warn: false,
    },
})]
#[case::interrupted_uses_poll(StepCase {
    attempt: LockAttempt::Interrupted,
    elapsed_ms: 10,
    warned: false,
    expected: LockStep::Retry {
        delay: Duration::from_millis(10),
        warn: false,
    },
})]
#[case::interrupted_clamps_to_remaining(StepCase {
    attempt: LockAttempt::Interrupted,
    elapsed_ms: 98,
    warned: false,
    expected: LockStep::Retry {
        delay: Duration::from_millis(2),
        warn: false,
    },
})]
fn next_step_matches_its_contract(#[case] case: StepCase) {
    let policy = policy().expect("valid policy should build");

    assert_eq!(
        policy.next_step(
            case.attempt,
            Duration::from_millis(case.elapsed_ms),
            case.warned,
        ),
        case.expected,
    );
}

#[test]
fn zero_timeout_is_rejected() {
    assert_eq!(
        LockPolicy::new(Duration::ZERO),
        Err(LockPolicyError::ZeroTimeout)
    );
}

#[test]
fn poll_interval_below_minimum_is_rejected() {
    let policy = LockPolicy::new(Duration::from_millis(50)).expect("timeout should be accepted");

    assert_eq!(
        policy.with_poll_interval(Duration::ZERO),
        Err(LockPolicyError::PollIntervalTooShort {
            minimum: MIN_POLL_INTERVAL,
        }),
    );
}

#[test]
fn minimum_poll_interval_is_accepted() {
    let policy = LockPolicy::new(Duration::from_millis(50))
        .and_then(|policy| policy.with_poll_interval(MIN_POLL_INTERVAL))
        .expect("minimum poll interval should be accepted");

    assert_eq!(policy.poll_interval(), MIN_POLL_INTERVAL);
}

#[test]
fn default_poll_interval_is_used_by_new() {
    let policy = LockPolicy::new(Duration::from_millis(50)).expect("timeout should be accepted");

    assert_eq!(policy.poll_interval(), DEFAULT_POLL_INTERVAL);
    assert_eq!(DEFAULT_POLL_INTERVAL, Duration::from_millis(25));
    assert_eq!(policy.timeout(), Duration::from_millis(50));
}

proptest! {
    #[test]
    fn simulated_acquisition_terminates_with_bounded_sleep_and_one_warning(
        timeout_ms in 1_u64..=2_000,
        poll_ms in 1_u64..=250,
        acquire_at in proptest::option::of(0_usize..64),
        attempts in proptest::collection::vec(any::<bool>(), 1..64),
    ) {
        let policy = LockPolicy::new(Duration::from_millis(timeout_ms))
            .expect("nonzero timeout should be accepted")
            .with_poll_interval(Duration::from_millis(poll_ms))
            .expect("nonzero poll interval should be accepted");
        let mut elapsed = Duration::ZERO;
        let mut sleeps = Duration::ZERO;
        let mut warnings = 0_u32;
        let mut steps = 0_usize;

        let acquired = loop {
            let attempt = if Some(steps) == acquire_at {
                LockAttempt::Acquired
            } else if attempts.get(steps).copied().unwrap_or(true) {
                LockAttempt::Contended
            } else {
                LockAttempt::Interrupted
            };

            match policy.next_step(attempt, elapsed, warnings > 0) {
                LockStep::Acquired => break true,
                LockStep::TimedOut { .. } => break false,
                LockStep::Retry { delay, warn } => {
                    prop_assert!(delay > Duration::ZERO);
                    prop_assert!(delay <= Duration::from_millis(poll_ms));
                    if warn {
                        warnings += 1;
                    }
                    sleeps = sleeps.checked_add(delay).expect("sleep total should not overflow");
                    elapsed = elapsed.checked_add(delay).expect("elapsed should not overflow");
                    steps += 1;
                    prop_assert!(steps < 10_000, "retry loop should terminate");
                }
                other => prop_assert!(false, "unexpected lock step: {other:?}"),
            }
        };

        prop_assert!(sleeps <= Duration::from_millis(timeout_ms));
        prop_assert!(warnings <= 1);
        if acquired {
            prop_assert!(elapsed < Duration::from_millis(timeout_ms));
        } else {
            prop_assert!(elapsed >= Duration::from_millis(timeout_ms));
        }
    }
}
