//! Assertions for expected clustering quality scores.

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
