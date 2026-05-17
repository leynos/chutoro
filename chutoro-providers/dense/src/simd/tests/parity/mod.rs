//! Property-based parity tests for dense SIMD backends.

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
