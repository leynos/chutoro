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

use chutoro_test_support::ci::property_test_profile::{PROPTEST_RNG_SEED, max_global_rejects_for};
use proptest::test_runner::{Config as ProptestConfig, RngSeed};

use crate::simd::{DensePointView, dispatch, kernels};

type PairwiseEntry = fn(&[f32], &[f32]) -> f32;
type QueryPointsEntry = fn(&[f32], &DensePointView<'_>, &mut [f32]);

/// Per-case execution bound for the forked weekly lane, in milliseconds.
///
/// proptest's default is zero, which disables the per-case timeout. The
/// measured mean per-case cost is 3.24 ms for the slowest property, so 20
/// seconds bounds a hung case with more than three orders of magnitude of
/// headroom. It also exceeds forked process startup, which proptest includes
/// in the bound.
pub(super) const FORKED_CASE_TIMEOUT_MS: u32 = 20_000;

/// Bound on one forked child's shrink phase, in milliseconds.
///
/// Shrinking only runs on failure. A minute leaves room for the fixed seed's
/// counterexample to shrink. proptest measures this bound from the start of
/// each `shrink()` call inside the child, so a candidate that hangs and
/// restarts the child starts a new minute; the 300-second nextest allowance
/// remains the outer bound across restarts, and an overrun is reported as a
/// timeout rather than as a counterexample. proptest's iteration bound
/// remains its default, four times the case count.
pub(super) const FORKED_SHRINK_TIME_MS: u32 = 60_000;

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
    parity_config(profile.cases(), profile.fork())
}

/// Builds the parity suite's configuration for a case count and run mode.
///
/// A non-zero per-case timeout implicitly enables forking, so the bound is
/// set only when the lane already forks. The pull-request and local lanes
/// stay unforked and keep proptest's disabled per-case timeout.
pub(super) fn parity_config(cases: u32, fork: bool) -> ProptestConfig {
    ProptestConfig {
        cases,
        fork,
        // Rejects are a budget for the whole run, so a deep run needs one in
        // proportion to the cases it asks for. See #260.
        max_global_rejects: max_global_rejects_for(cases),
        timeout: if fork { FORKED_CASE_TIMEOUT_MS } else { 0 },
        max_shrink_time: if fork { FORKED_SHRINK_TIME_MS } else { 0 },
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
