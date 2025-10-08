use crate::distance::helpers::{accumulate_components, validate_dimensions};
use crate::distance::types::{CosineNorms, Distance, Norm, Result, Vector, VectorKind};

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
/// - [`DistanceError::ZeroLength`] when any input is empty.
/// - [`DistanceError::DimensionMismatch`] when input lengths differ.
/// - [`DistanceError::NonFinite`] when a value is NaN or infinite.
/// - [`DistanceError::ZeroMagnitude`] when either vector has zero L2 norm.
/// - [`DistanceError::InvalidNorm`] when pre-computed norms are non-finite.
pub fn cosine_distance(
    left: &[f32],
    right: &[f32],
    norms: Option<CosineNorms>,
) -> Result<Distance> {
    let left = Vector::new(left, VectorKind::Left)?;
    let right = Vector::new(right, VectorKind::Right)?;
    validate_dimensions(&left, &right)?;

    let (dot, left_squares, right_squares) = accumulate_components(&left, &right);

    let (left_norm, right_norm) = match norms {
        Some(norms) => {
            let _ = Norm::from_squared_sum(left_squares, VectorKind::Left)?;
            let _ = Norm::from_squared_sum(right_squares, VectorKind::Right)?;
            (norms.left_norm(), norms.right_norm())
        }
        None => (
            Norm::from_squared_sum(left_squares, VectorKind::Left)?,
            Norm::from_squared_sum(right_squares, VectorKind::Right)?,
        ),
    };

    let denominator = f64::from(*left_norm) * f64::from(*right_norm);
    let similarity = (dot / denominator) as f32;
    // Theoretical range is [-1, 1], but numerical noise can spill over.
    let similarity = similarity.clamp(-1.0, 1.0);

    Ok(Distance::from_raw(1.0 - similarity))
}
