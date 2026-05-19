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

fn pairwise_entries() -> Vec<(dispatch::EuclideanBackend, PairwiseEntry)> {
    dispatch::enabled_backends()
        .into_iter()
        .filter_map(|backend| kernels::pairwise_entry(backend).map(|entry| (backend, entry)))
        .collect()
}

fn query_points_entries() -> Vec<(dispatch::EuclideanBackend, QueryPointsEntry)> {
    dispatch::enabled_backends()
        .into_iter()
        .filter_map(|backend| kernels::query_points_entry(backend).map(|entry| (backend, entry)))
        .collect()
}
