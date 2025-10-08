use super::types::{DistanceError, Result, Vector};

/// Ensures both vectors share the same dimensionality.
pub(crate) fn validate_dimensions(left: &Vector<'_>, right: &Vector<'_>) -> Result<()> {
    if left.dimension() != right.dimension() {
        return Err(DistanceError::DimensionMismatch {
            left: left.dimension(),
            right: right.dimension(),
        });
    }
    Ok(())
}

/// Accumulates the dot product and squared magnitudes across both vectors.
pub(crate) fn accumulate_components(left: &Vector<'_>, right: &Vector<'_>) -> (f64, f64, f64) {
    let mut dot = 0.0f64;
    let mut left_squares = 0.0f64;
    let mut right_squares = 0.0f64;

    for (&l, &r) in left.iter().zip(right.iter()) {
        dot += f64::from(l) * f64::from(r);
        left_squares += f64::from(l) * f64::from(l);
        right_squares += f64::from(r) * f64::from(r);
    }

    (dot, left_squares, right_squares)
}
