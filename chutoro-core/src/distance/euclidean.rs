//! Euclidean distance implementation for validated vectors.

use core::ops::{AddAssign, Mul, Sub};

use crate::distance::helpers::validate_dimensions;
use crate::distance::types::{Distance, Result, Vector, VectorKind, narrow_to_f32};

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
/// - [`crate::distance::DistanceError::ZeroLength`] when any input is empty.
/// - [`crate::distance::DistanceError::DimensionMismatch`]
///   when input lengths differ.
/// - [`crate::distance::DistanceError::NonFinite`] when a value is NaN or
///   infinite.
pub fn euclidean_distance(left_values: &[f32], right_values: &[f32]) -> Result<Distance> {
    let left_vector = Vector::new(left_values, VectorKind::Left)?;
    let right_vector = Vector::new(right_values, VectorKind::Right)?;
    validate_dimensions(&left_vector, &right_vector)?;

    let mut sum = 0.0f64;
    for (&l, &r) in left_vector.iter().zip(right_vector.iter()) {
        let difference = f64::from(l).sub(f64::from(r));
        sum.add_assign(difference.mul(difference));
    }

    Ok(Distance::from_raw(narrow_to_f32(sum.sqrt())))
}
