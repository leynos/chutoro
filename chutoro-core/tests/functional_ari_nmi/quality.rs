//! Quality assertions shared by the functional clustering tests.
//!
//! `assert_quality_score_is_one` keeps the absolute tolerance for scores
//! derived from floating-point calculations in one place, so every functional
//! ARI/NMI case compares against the same bound. This is a private child of
//! the `functional_ari_nmi` integration-test root, which declares it with
//! `#[path]`.

/// Absolute tolerance for clustering-quality values derived from floating-point calculations.
const QUALITY_SCORE_TOLERANCE: f64 = 1.0e-12_f64;

#[expect(
    clippy::float_arithmetic,
    reason = "derived clustering-quality checks need an absolute delta"
)]
pub(super) fn assert_quality_score_is_one(score: f64) {
    let delta = (score - 1.0_f64).abs();
    assert!(
        delta <= QUALITY_SCORE_TOLERANCE,
        "score={score}, delta={delta}, tolerance={QUALITY_SCORE_TOLERANCE}"
    );
}
