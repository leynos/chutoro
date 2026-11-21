//! Dense SIMD parity timeout contracts.
//!
//! The weekly and manual property lane runs the dense parity suite at 25,000
//! forked cases. These tests pin the overrides that bound it and the profile
//! precedence that lets the heavy lane raise the allowance, so a configuration
//! edit cannot silently return the suite to the 60-second base (#264).

use rstest::rstest;

use super::{NEXTEST_CONFIG, has_override_with_fragments, override_blocks};

const DENSE_PARITY_FILTER: &str =
    "filter = \"package(chutoro-providers-dense) & test(/simd::tests::parity::/)\"";

/// Returns a profile's dense parity period in seconds.
///
/// The period is read from the override block carrying the dense filter, so an
/// unrelated override's period cannot satisfy the reading.
fn dense_parity_period(profile_name: &str) -> Option<u64> {
    let block = override_blocks(profile_name)
        .into_iter()
        .find(|block| block.contains(DENSE_PARITY_FILTER))?;
    let (_, rest) = block.split_once("period = \"")?;
    rest.split_once('s')?.0.parse().ok()
}

/// Verifies every profile that runs the suite terminates a slow case.
#[rstest]
#[case("default")]
#[case("ci")]
#[case("property-heavy")]
fn dense_parity_profiles_declare_a_bounded_timeout(#[case] profile_name: &str) {
    assert_override_present!(
        profile_name,
        [DENSE_PARITY_FILTER, "terminate-after = 1", "period = \""]
    );
    assert!(
        dense_parity_period(profile_name).is_some(),
        "profile '{profile_name}' should declare a whole-second dense parity period"
    );
}

/// Verifies the heavy profile layers over the CI profile rather than beside it.
#[test]
fn property_heavy_profile_inherits_ci() {
    assert!(
        NEXTEST_CONFIG.contains("[profile.property-heavy]\ninherits = \"ci\""),
        "the heavy profile must inherit `ci`, or the weekly lane would lose its \
         thread count and setup script"
    );
}

/// Verifies the heavy allowance is wider than the CI one it replaces.
#[test]
fn property_heavy_dense_timeout_exceeds_the_ci_allowance() {
    let ci = dense_parity_period("ci").expect("the CI profile must bound dense parity");
    let heavy =
        dense_parity_period("property-heavy").expect("the heavy profile must bound dense parity");
    assert!(
        heavy > ci,
        "the heavy dense allowance ({heavy}s) must exceed the CI allowance ({ci}s)"
    );
}
