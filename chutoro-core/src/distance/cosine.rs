//! Cosine distance implementation built on validated vector primitives.

use core::ops::{Div, Mul, Sub};

use crate::distance::helpers::{
    accumulate_components, ensure_cached_norms_usable, validate_dimensions,
};
use crate::distance::types::{
    CosineNorms, Distance, Norm, Result, Vector, VectorKind, narrow_to_f32,
};

/// Computes the cosine distance between two vectors.
///
/// The optional [`CosineNorms`] parameter allows callers to reuse pre-computed
/// L2 norms and avoid recomputing them for every query.
///
/// # Examples
///
/// ```
/// use chutoro_core::{CosineNorms, DistanceError, cosine_distance};
///
/// fn main() -> Result<(), DistanceError> {
///     let a = [1.0f32, 0.0, 0.0];
///     let b = [0.0f32, 1.0, 0.0];
///
///     // Compute norms on the fly.
///     let orthogonal = cosine_distance(&a, &b, None)?;
///     assert!((orthogonal.value() - 1.0).abs() < 1e-6);
///
///     // Reuse pre-computed norms.
///     let norms = CosineNorms::from_vectors(&a, &b)?;
///     let again = cosine_distance(&a, &b, Some(norms))?;
///     assert!((again.value() - 1.0).abs() < 1e-6);
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
/// - [`crate::distance::DistanceError::ZeroMagnitude`] when either vector has
///   zero L2 norm.
/// - [`crate::distance::DistanceError::InvalidNorm`] when pre-computed norms
///   are non-finite.
pub fn cosine_distance(
    left_values: &[f32],
    right_values: &[f32],
    cached_norms: Option<CosineNorms>,
) -> Result<Distance> {
    let left_vector = Vector::new(left_values, VectorKind::Left)?;
    let right_vector = Vector::new(right_values, VectorKind::Right)?;
    validate_dimensions(&left_vector, &right_vector)?;

    let (dot, left_squares, right_squares) = accumulate_components(&left_vector, &right_vector);

    let (left_norm, right_norm) = match cached_norms {
        Some(precomputed_norms) => {
            ensure_cached_norms_usable(left_squares, right_squares)?;
            (
                precomputed_norms.left_norm(),
                precomputed_norms.right_norm(),
            )
        }
        None => (
            Norm::from_squared_sum(left_squares, VectorKind::Left)?,
            Norm::from_squared_sum(right_squares, VectorKind::Right)?,
        ),
    };

    let denominator = f64::from(*left_norm).mul(f64::from(*right_norm));
    let similarity = narrow_to_f32(dot.div(denominator));
    // Theoretical range is [-1, 1], but numerical noise can spill over.
    let clamped_similarity = similarity.clamp(-1.0, 1.0);

    Ok(Distance::from_raw(1.0_f32.sub(clamped_similarity)))
}
