//! `ef_construction` sweep constants and parameter helpers.
//!
//! Provides the parameter matrix for the `hnsw_build_ef_sweep` benchmark
//! group, which varies `ef_construction` independently of `M` to reveal
//! build-time versus recall trade-offs.

use chutoro_core::HnswParams;

use crate::error::BenchSetupError;

/// Dataset sizes used for the `ef_construction` sweep benchmarks.
///
/// A representative small and large size from the main sweep, chosen to
/// keep total case count manageable while still showing scaling effects.
pub const EF_SWEEP_POINT_COUNTS: &[usize] = &[500, 5_000];

/// HNSW M (`max_connections`) values used for the `ef_construction` sweep.
///
/// The extremes of the main benchmark M range ({8, 24}) show how
/// `ef_construction` interacts with graph connectivity density.
pub const EF_SWEEP_MAX_CONNECTIONS: &[usize] = &[8, 24];

/// `ef_construction` values to sweep.
///
/// `0` is a sentinel meaning "use the M\*2 default" (matching the baseline
/// used by the existing benchmark groups). The three explicit values
/// (`100`, `200`, `400`) follow the roadmap guidance for showing
/// diminishing-returns behaviour at increasing construction search widths.
pub const EF_CONSTRUCTION_VALUES: &[usize] = &[0, 100, 200, 400];

/// Resolves the `ef_construction` sentinel value.
///
/// When `ef_raw` is `0`, returns `m * 2` (the baseline default). Otherwise
/// returns `ef_raw` unchanged.
///
/// # Examples
///
/// ```
/// use chutoro_benches::ef_sweep::resolve_ef_construction;
/// assert_eq!(resolve_ef_construction(8, 0), 16);
/// assert_eq!(resolve_ef_construction(24, 100), 100);
/// ```
#[must_use]
pub const fn resolve_ef_construction(m: usize, ef_raw: usize) -> usize {
    if ef_raw == 0 {
        m.saturating_mul(2)
    } else {
        ef_raw
    }
}

/// Creates [`HnswParams`] with an explicit `ef_construction` and seed.
///
/// # Errors
///
/// Returns [`BenchSetupError::Hnsw`] when `ef_construction < m` or `m == 0`.
///
/// # Examples
///
/// ```
/// use chutoro_benches::ef_sweep::make_hnsw_params_with_ef;
/// let params = make_hnsw_params_with_ef(16, 200, 42)
///     .expect("valid parameters");
/// assert_eq!(params.max_connections(), 16);
/// assert_eq!(params.ef_construction(), 200);
/// ```
pub fn make_hnsw_params_with_ef(
    m: usize,
    ef_construction: usize,
    seed: u64,
) -> Result<HnswParams, BenchSetupError> {
    Ok(HnswParams::new(m, ef_construction)?.with_rng_seed(seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    // -- resolve_ef_construction ----------------------------------------

    #[rstest]
    #[case::sentinel_m8(8, 0, 16)]
    #[case::sentinel_m24(24, 0, 48)]
    #[case::passthrough_100(8, 100, 100)]
    #[case::passthrough_400(24, 400, 400)]
    fn resolve_ef_construction_returns_expected(
        #[case] m: usize,
        #[case] ef_raw: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(resolve_ef_construction(m, ef_raw), expected);
    }

    // -- make_hnsw_params_with_ef: happy paths --------------------------

    #[rstest]
    #[case::boundary_m24_ef48(24, 48)]
    #[case::high_ef_m8_ef400(8, 400)]
    #[case::equal_m_ef(16, 16)]
    #[case::typical_m16_ef200(16, 200)]
    fn make_params_accepts_valid_combinations(#[case] m: usize, #[case] ef: usize) {
        let params = make_hnsw_params_with_ef(m, ef, 42).expect("valid (m, ef) pair must succeed");
        assert_eq!(params.max_connections(), m);
        assert_eq!(params.ef_construction(), ef);
    }

    // -- make_hnsw_params_with_ef: unhappy paths ------------------------

    #[rstest]
    #[case::ef_below_m(24, 10)]
    #[case::zero_m(0, 100)]
    fn make_params_rejects_invalid_combinations(#[case] m: usize, #[case] ef: usize) {
        assert!(make_hnsw_params_with_ef(m, ef, 42).is_err());
    }
}
