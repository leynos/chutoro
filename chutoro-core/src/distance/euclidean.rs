use crate::distance::helpers::validate_dimensions;
use crate::distance::types::{Distance, Result, Vector, VectorKind};

/// Computes the Euclidean distance between two vectors.
///
/// # Examples
///
/// ```
/// use chutoro_core::{DistanceError, euclidean_distance};
///
/// fn main() -> Result<(), DistanceError> {
///     let distance = euclidean_distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 8.0])?;
///     assert!((distance.value() - 7.071_068).abs() < 1e-6);
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// - [`DistanceError::ZeroLength`] when any input is empty.
/// - [`DistanceError::DimensionMismatch`] when input lengths differ.
/// - [`DistanceError::NonFinite`] when a value is NaN or infinite.
pub fn euclidean_distance(left: &[f32], right: &[f32]) -> Result<Distance> {
    let left = Vector::new(left, VectorKind::Left)?;
    let right = Vector::new(right, VectorKind::Right)?;
    validate_dimensions(&left, &right)?;

    let mut sum = 0.0f64;
    for (&l, &r) in left.iter().zip(right.iter()) {
        let diff = f64::from(l) - f64::from(r);
        sum += diff * diff;
    }

    Ok(Distance::from_raw(sum.sqrt() as f32))
}
