//! Tests for the parity suite's proptest configuration.
//!
//! These sit outside `parity` on purpose. Both property lanes filter on
//! `simd::tests::parity::` and set `PROPTEST_CASES`, so a configuration test
//! living there would be run under an overridden case count, which is exactly
//! what its depth guard refuses to accept. This is a test of how the config
//! is built, not of what the kernels compute.

use super::parity::proptest_config;
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
