//! Tests for the parity suite's proptest configuration.
//!
//! These sit outside `parity` on purpose. Both property lanes filter on
//! `simd::tests::parity::` and set `PROPTEST_CASES`, so a configuration test
//! living there would be run under an overridden case count, which is exactly
//! what its depth guard refuses to accept. This is a test of how the config
//! is built, not of what the kernels compute.

use super::parity::{
    FORKED_CASE_TIMEOUT_MS, FORKED_SHRINK_TIME_MS, parity_config, proptest_config,
};
use chutoro_test_support::ci::property_test_profile::{
    DEFAULT_MAX_GLOBAL_REJECTS, max_global_rejects_for,
};

/// The parity suite config sizes its reject budget from its case count.
///
/// These suites already fork and run under the weekly profile, so the
/// flat 1024 default is the next thing they would hit as their case
/// counts rise (#260).
#[test]
fn the_parity_config_scales_its_reject_budget() {
    let config = proptest_config(25_000);

    assert!(
        max_global_rejects_for(config.cases) > DEFAULT_MAX_GLOBAL_REJECTS,
        "the fixture must ask for a run deep enough that the floor is not \
         the answer, or this test passes whether the budget is derived or \
         left on proptest's default"
    );

    assert_eq!(
        config.max_global_rejects,
        max_global_rejects_for(config.cases),
        "a deep run left on proptest's flat default aborts before it finishes"
    );
}

/// Verifies the forked lane carries an explicit per-case timeout and shrink
/// bound.
///
/// proptest's defaults leave the per-case timeout and shrink time at zero,
/// which disables both. The weekly lane therefore had no in-case bound at
/// all: the 300-second nextest allowance bounded the whole property, so one
/// stuck case could consume it without proptest reporting which case hung.
#[test]
fn the_forked_lane_bounds_each_case_and_its_shrinking() {
    let config = parity_config(25_000, true);

    assert_eq!(
        config.timeout, FORKED_CASE_TIMEOUT_MS,
        "a forked run must bound each case rather than inherit proptest's \
         zero, which disables the timeout"
    );
    assert_eq!(
        config.max_shrink_time, FORKED_SHRINK_TIME_MS,
        "a forked run must bound shrinking rather than inherit proptest's \
         zero, which never gives up"
    );
    assert_eq!(
        config.max_shrink_iters,
        proptest::test_runner::Config::default().max_shrink_iters,
        "the iteration bound stays proptest's default of four times the cases"
    );
}

/// Verifies an unforked lane keeps proptest's unbounded defaults.
///
/// A non-zero per-case timeout implicitly enables forking, so setting it on
/// the 250-case pull-request lane would silently change how that lane runs.
#[test]
fn the_unforked_lanes_keep_proptest_defaults() {
    let config = parity_config(250, false);

    assert!(!config.fork, "the pull-request lane stays unforked");
    assert_eq!(
        config.timeout, 0,
        "a non-zero timeout would implicitly enable forking"
    );
    assert_eq!(config.max_shrink_time, 0);
}

/// Verifies the case bound leaves the measured per-case cost ample headroom.
///
/// The slowest measured property averages 3.24 ms per case including fork
/// overhead. A bound below 100 times that would turn a slow runner into a
/// spurious per-case failure rather than a reported timeout.
#[test]
fn the_case_bound_exceeds_the_measured_case_cost() {
    const MEASURED_MEAN_CASE_MS: f64 = 3.24;

    assert!(
        f64::from(FORKED_CASE_TIMEOUT_MS) > 100.0 * MEASURED_MEAN_CASE_MS,
        "a {FORKED_CASE_TIMEOUT_MS} ms per-case bound leaves too little room \
         over the measured {MEASURED_MEAN_CASE_MS} ms mean case"
    );
}
