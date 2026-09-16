//! Property-based parity tests for dense SIMD Euclidean backends.
//!
//! These tests use `proptest` to generate random dense inputs, then verify
//! that every enabled backend agrees with the scalar oracle within the
//! [`DistanceSemantics`](crate::simd::semantics::DistanceSemantics) epsilon.
//! In this context, parity means each compiled and runtime-available SIMD
//! implementation returns the same observable distance semantics as the scalar
//! implementation for the same generated fixture.
//!
//! The submodules divide the suite by fixture shape and policy surface:
//! `strategies` owns input generators, `pairwise` checks finite pairwise
//! distances, `query_points` checks finite query-to-points batches, and
//! `non_finite` checks NaN-canonicalization behaviour.
//!
//! `pairwise_entries` and `query_points_entries` combine
//! [`dispatch::enabled_backends`] with [`kernels::pairwise_entry`] and
//! [`kernels::query_points_entry`]. That keeps the suite limited to backends
//! that are both compiled into the current build and available on the current
//! CPU, so feature flags and runtime detection automatically shape the test
//! set.
//!
//! To add a backend to the parity suite, implement its entry point in
//! `kernels.rs`, add it to [`dispatch::enabled_backends`], and the helpers in
//! this module will pick it up without another test-specific dispatch table.

mod non_finite;
mod pairwise;
mod query_points;
mod strategies;

use chutoro_test_support::ci::property_test_profile::PROPTEST_RNG_SEED;
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

use crate::simd::{DensePointView, dispatch, kernels};

type PairwiseEntry = fn(&[f32], &[f32]) -> f32;
type QueryPointsEntry = fn(&[f32], &DensePointView<'_>, &mut [f32]);

/// Builds a CI-tuned proptest configuration for parity properties.
///
/// The run profile is loaded from
/// `chutoro_test_support::ci::property_test_profile`; `default_cases` is used
/// when no CI override is present. The returned [`ProptestConfig`] carries the
/// selected case count and fork flag.
pub(super) fn proptest_config(default_cases: u32) -> ProptestConfig {
    const DEFAULT_FORK: bool = false;

    let profile = chutoro_test_support::ci::property_test_profile::ProptestRunProfile::load(
        default_cases,
        DEFAULT_FORK,
    );
    ProptestConfig {
        cases: profile.cases(),
        fork: profile.fork(),
        // Rejects are a budget for the whole run, so a deep run needs one in
        // proportion to the cases it asks for. See #260.
        max_global_rejects: profile.max_global_rejects(),
        rng_seed: RngSeed::Fixed(PROPTEST_RNG_SEED),
        ..ProptestConfig::default()
    }
}

/// Enumerates pairwise kernels for every backend available to this process.
///
/// Backends come from [`dispatch::enabled_backends`], then
/// [`kernels::pairwise_entry`] resolves each compiled-and-runtime-available
/// backend to its pairwise entry point before the pairs are collected. Missing
/// entry points are programming errors, reported as `Err` so the calling test
/// fails instead of skipping the backend.
fn pairwise_entries() -> Result<Vec<(dispatch::EuclideanBackend, PairwiseEntry)>, String> {
    dispatch::enabled_backends()
        .into_iter()
        .map(|backend| {
            let entry = kernels::pairwise_entry(backend).ok_or_else(|| {
                format!("enabled backend {backend:?} must have a pairwise kernel entrypoint")
            })?;
            Ok((backend, entry))
        })
        .collect()
}

/// Enumerates query-to-points kernels for every backend available here.
///
/// Backends come from [`dispatch::enabled_backends`], then
/// [`kernels::query_points_entry`] resolves each
/// compiled-and-runtime-available backend to its query-to-points entry point
/// before the pairs are collected. Missing entry points are programming
/// errors, reported as `Err` so the calling test fails instead of skipping
/// the backend.
fn query_points_entries() -> Result<Vec<(dispatch::EuclideanBackend, QueryPointsEntry)>, String> {
    dispatch::enabled_backends()
        .into_iter()
        .map(|backend| {
            let entry = kernels::query_points_entry(backend).ok_or_else(|| {
                format!("enabled backend {backend:?} must have a query-points kernel entrypoint")
            })?;
            Ok((backend, entry))
        })
        .collect()
}

#[cfg(test)]
mod config_tests {
    //! Unit tests for the parity suite configuration.

    use super::proptest_config;
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
}
