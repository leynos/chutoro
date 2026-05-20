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
//! `non_finite` checks NaN-canonicalisation behaviour.
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

use proptest::test_runner::Config as ProptestConfig;

use crate::simd::{DensePointView, dispatch, kernels};

type PairwiseEntry = fn(&[f32], &[f32]) -> f32;
type QueryPointsEntry = fn(&[f32], &DensePointView<'_>, &mut [f32]);

/// Builds a CI-tuned proptest configuration for parity properties.
///
/// The run profile is loaded from
/// `chutoro_test_support::ci::property_test_profile`; `default_cases` is used
/// when no CI override is present. The returned [`ProptestConfig`] carries the
/// selected case count and fork flag.
fn proptest_config(default_cases: u32) -> ProptestConfig {
    const DEFAULT_FORK: bool = false;

    let profile = chutoro_test_support::ci::property_test_profile::ProptestRunProfile::load(
        default_cases,
        DEFAULT_FORK,
    );
    ProptestConfig {
        cases: profile.cases(),
        fork: profile.fork(),
        ..ProptestConfig::default()
    }
}

/// Enumerates pairwise kernels for every backend available to this process.
///
/// Backends come from [`dispatch::enabled_backends`], then
/// [`kernels::pairwise_entry`] resolves each compiled-and-runtime-available
/// backend to its pairwise entry point before the pairs are collected. Missing
/// entry points are programming errors and panic instead of being skipped.
fn pairwise_entries() -> Vec<(dispatch::EuclideanBackend, PairwiseEntry)> {
    dispatch::enabled_backends()
        .into_iter()
        .map(|backend| {
            let entry = kernels::pairwise_entry(backend)
                .expect("enabled backend must have a pairwise kernel entrypoint");
            (backend, entry)
        })
        .collect()
}

/// Enumerates query-to-points kernels for every backend available here.
///
/// Backends come from [`dispatch::enabled_backends`], then
/// [`kernels::query_points_entry`] resolves each
/// compiled-and-runtime-available backend to its query-to-points entry point
/// before the pairs are collected. Missing entry points are programming errors
/// and panic instead of being skipped.
fn query_points_entries() -> Vec<(dispatch::EuclideanBackend, QueryPointsEntry)> {
    dispatch::enabled_backends()
        .into_iter()
        .map(|backend| {
            let entry = kernels::query_points_entry(backend)
                .expect("enabled backend must have a query-points kernel entrypoint");
            (backend, entry)
        })
        .collect()
}
