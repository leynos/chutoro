//! Distance primitives for built-in numeric metrics.
//!
//! The walking skeleton exposes scalar implementations for Euclidean and
//! cosine distances. These routines validate their inputs and surface detailed
//! errors so callers can react appropriately during ingestion or algorithmic
//! execution.

use core::fmt;

use thiserror::Error;

/// Identifies whether an error was produced while inspecting the left or right
/// vector argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorKind {
    /// Value originating from the first argument.
    Left,
    /// Value originating from the second argument.
    Right,
}

impl fmt::Display for VectorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => f.write_str("left"),
            Self::Right => f.write_str("right"),
        }
    }
}

/// Errors emitted while computing distances.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DistanceError {
    /// Either input vector had zero length.
    #[error("vectors must have positive dimension")]
    ZeroLength,
    /// Input vectors had different lengths.
    #[error("dimension mismatch: left={left}, right={right}")]
    DimensionMismatch { left: usize, right: usize },
    /// Encountered a non-finite value in one of the vectors.
    #[error("{which} vector contains a non-finite value at index {index}: {value}")]
    NonFinite {
        which: VectorKind,
        index: usize,
        value: f32,
    },
    /// Cosine distance is undefined for zero-magnitude vectors.
    #[error("{which} vector has zero magnitude")]
    ZeroMagnitude { which: VectorKind },
    /// Provided norms must be finite.
    #[error("provided {which} norm must be finite (got {value})")]
    InvalidNorm { which: VectorKind, value: f32 },
}

/// Convenient alias for distance computations.
pub type Result<T> = core::result::Result<T, DistanceError>;

/// Pre-computed L2 norms for cosine distance calculations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CosineNorms {
    left: f32,
    right: f32,
}

impl CosineNorms {
    /// Builds [`CosineNorms`] from explicit norm values.
    ///
    /// # Errors
    ///
    /// Returns [`DistanceError::InvalidNorm`] when a norm is non-finite and
    /// [`DistanceError::ZeroMagnitude`] when a norm is not strictly positive.
    pub fn new(left: f32, right: f32) -> Result<Self> {
        let left = validate_norm(left, VectorKind::Left)?;
        let right = validate_norm(right, VectorKind::Right)?;
        Ok(Self { left, right })
    }

    /// Computes norms from the provided vectors.
    ///
    /// # Errors
    ///
    /// Propagates validation errors surfaced by [`cosine_distance`].
    pub fn from_vectors(left: &[f32], right: &[f32]) -> Result<Self> {
        validate_lengths(left, right)?;
        let left = compute_norm(left, VectorKind::Left)?;
        let right = compute_norm(right, VectorKind::Right)?;
        Ok(Self { left, right })
    }

    /// Returns the stored norm for the left vector.
    #[must_use]
    pub fn left(&self) -> f32 {
        self.left
    }

    /// Returns the stored norm for the right vector.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.right
    }
}

/// Computes the Euclidean distance between two vectors.
///
/// # Examples
///
/// ```
/// use chutoro_core::{euclidean_distance, DistanceError};
///
/// fn main() -> Result<(), DistanceError> {
///     let distance = euclidean_distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 8.0])?;
///     assert!((distance - 7.071_068).abs() < 1e-6);
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// - [`DistanceError::ZeroLength`] when any input is empty.
/// - [`DistanceError::DimensionMismatch`] when input lengths differ.
/// - [`DistanceError::NonFinite`] when a value is NaN or infinite.
pub fn euclidean_distance(left: &[f32], right: &[f32]) -> Result<f32> {
    validate_lengths(left, right)?;

    let mut sum = 0.0f64;
    for (index, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        ensure_finite(l, VectorKind::Left, index)?;
        ensure_finite(r, VectorKind::Right, index)?;

        let diff = f64::from(l) - f64::from(r);
        sum += diff * diff;
    }

    Ok((sum).sqrt() as f32)
}

/// Computes the cosine distance between two vectors.
///
/// The optional [`CosineNorms`] parameter allows callers to reuse pre-computed
/// L2 norms and avoid recomputing them for every query.
///
/// # Examples
///
/// ```
/// use chutoro_core::{cosine_distance, CosineNorms, DistanceError};
///
/// fn main() -> Result<(), DistanceError> {
///     let a = [1.0f32, 0.0, 0.0];
///     let b = [0.0f32, 1.0, 0.0];
///
///     // Compute norms on the fly.
///     let orthogonal = cosine_distance(&a, &b, None)?;
///     assert!((orthogonal - 1.0).abs() < 1e-6);
///
///     // Reuse pre-computed norms.
///     let norms = CosineNorms::from_vectors(&a, &b)?;
///     let again = cosine_distance(&a, &b, Some(norms))?;
///     assert!((again - 1.0).abs() < 1e-6);
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
pub fn cosine_distance(left: &[f32], right: &[f32], norms: Option<CosineNorms>) -> Result<f32> {
    validate_lengths(left, right)?;

    let mut dot = 0.0f64;
    let mut left_squares = 0.0f64;
    let mut right_squares = 0.0f64;
    let mut left_has_magnitude = false;
    let mut right_has_magnitude = false;

    for (index, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        ensure_finite(l, VectorKind::Left, index)?;
        ensure_finite(r, VectorKind::Right, index)?;
        dot += f64::from(l) * f64::from(r);
        left_has_magnitude |= l != 0.0;
        right_has_magnitude |= r != 0.0;

        if norms.is_none() {
            left_squares += f64::from(l) * f64::from(l);
            right_squares += f64::from(r) * f64::from(r);
        }
    }

    let (left_norm, right_norm) = match norms {
        Some(norms) => {
            if !left_has_magnitude {
                return Err(DistanceError::ZeroMagnitude {
                    which: VectorKind::Left,
                });
            }
            if !right_has_magnitude {
                return Err(DistanceError::ZeroMagnitude {
                    which: VectorKind::Right,
                });
            }
            (norms.left, norms.right)
        }
        None => (
            finalise_norm(left_squares, VectorKind::Left)?,
            finalise_norm(right_squares, VectorKind::Right)?,
        ),
    };

    let left_norm = validate_norm(left_norm, VectorKind::Left)?;
    let right_norm = validate_norm(right_norm, VectorKind::Right)?;

    let denominator = f64::from(left_norm) * f64::from(right_norm);
    let similarity = (dot / denominator) as f32;
    // Theoretical range is [-1, 1], but numerical noise can spill over.
    let similarity = similarity.clamp(-1.0, 1.0);

    Ok(1.0 - similarity)
}

fn validate_lengths(left: &[f32], right: &[f32]) -> Result<()> {
    if left.is_empty() || right.is_empty() {
        return Err(DistanceError::ZeroLength);
    }
    if left.len() != right.len() {
        return Err(DistanceError::DimensionMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    Ok(())
}

fn ensure_finite(value: f32, which: VectorKind, index: usize) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DistanceError::NonFinite {
            which,
            index,
            value,
        })
    }
}

fn compute_norm(values: &[f32], which: VectorKind) -> Result<f32> {
    let mut sum = 0.0f64;
    for (index, value) in values.iter().enumerate() {
        ensure_finite(*value, which, index)?;
        sum += f64::from(*value) * f64::from(*value);
    }

    finalise_norm(sum, which)
}

fn finalise_norm(sum_of_squares: f64, which: VectorKind) -> Result<f32> {
    let norm = sum_of_squares.sqrt() as f32;
    if norm == 0.0 {
        Err(DistanceError::ZeroMagnitude { which })
    } else {
        Ok(norm)
    }
}

fn validate_norm(norm: f32, which: VectorKind) -> Result<f32> {
    if !norm.is_finite() {
        return Err(DistanceError::InvalidNorm { which, value: norm });
    }
    if norm == 0.0 {
        return Err(DistanceError::ZeroMagnitude { which });
    }
    if norm < 0.0 {
        return Err(DistanceError::InvalidNorm { which, value: norm });
    }
    Ok(norm)
}
