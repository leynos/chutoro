//! Distance primitives for built-in numeric metrics.
//!
//! The walking skeleton exposes scalar implementations for Euclidean and
//! cosine distances. These routines validate their inputs and surface detailed
//! errors so callers can react appropriately during ingestion or algorithmic
//! execution.

use core::{fmt, ops::Deref};

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

/// Vector newtype that validates dimensionality and finiteness on construction.
#[derive(Clone, Copy, Debug)]
pub struct Vector<'a>(&'a [f32]);

impl<'a> Vector<'a> {
    /// Validates and constructs a [`Vector`].
    ///
    /// # Errors
    ///
    /// Returns [`DistanceError::ZeroLength`] when the slice is empty and
    /// [`DistanceError::NonFinite`] when a value is NaN or infinite.
    pub fn new(values: &'a [f32], which: VectorKind) -> Result<Self> {
        if values.is_empty() {
            return Err(DistanceError::ZeroLength);
        }

        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DistanceError::NonFinite {
                    which,
                    index,
                    value: *value,
                });
            }
        }

        Ok(Self(values))
    }

    /// Returns the dimensionality of the vector.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.0.len()
    }
}

impl<'a> AsRef<[f32]> for Vector<'a> {
    fn as_ref(&self) -> &[f32] {
        self.0
    }
}

impl<'a> Deref for Vector<'a> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

/// Validated L2 norm for cosine distance calculations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Norm(f32);

impl Norm {
    /// Validates an explicit norm value.
    ///
    /// # Errors
    ///
    /// Returns [`DistanceError::InvalidNorm`] when the value is non-finite or
    /// negative, and [`DistanceError::ZeroMagnitude`] when the value is zero.
    pub fn new(value: f32, which: VectorKind) -> Result<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(DistanceError::InvalidNorm { which, value });
        }
        if value == 0.0 {
            return Err(DistanceError::ZeroMagnitude { which });
        }
        Ok(Self(value))
    }

    /// Computes an L2 norm from a validated vector.
    ///
    /// # Errors
    ///
    /// Propagates [`DistanceError::ZeroMagnitude`] when the vector has zero
    /// length in terms of magnitude.
    pub fn from_vector(vector: &Vector<'_>, which: VectorKind) -> Result<Self> {
        let mut sum = 0.0f64;
        for value in vector.iter() {
            sum += f64::from(*value) * f64::from(*value);
        }

        Self::new(sum.sqrt() as f32, which)
    }

    /// Returns the validated norm value.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Deref for Norm {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Distance result newtype.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distance(f32);

impl Distance {
    fn from_raw(value: f32) -> Self {
        Self(value)
    }

    /// Returns the raw distance value.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Deref for Distance {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Distance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Pre-computed L2 norms for cosine distance calculations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CosineNorms {
    left: Norm,
    right: Norm,
}

impl CosineNorms {
    /// Builds [`CosineNorms`] from explicit norm values.
    ///
    /// # Errors
    ///
    /// Returns [`DistanceError::InvalidNorm`] when a norm is non-finite or
    /// negative and [`DistanceError::ZeroMagnitude`] when a norm is zero.
    pub fn new(left: f32, right: f32) -> Result<Self> {
        let left = Norm::new(left, VectorKind::Left)?;
        let right = Norm::new(right, VectorKind::Right)?;
        Ok(Self { left, right })
    }

    /// Computes norms from the provided vectors.
    ///
    /// # Errors
    ///
    /// Propagates validation errors surfaced by [`cosine_distance`].
    pub fn from_vectors(left: &[f32], right: &[f32]) -> Result<Self> {
        let left = Vector::new(left, VectorKind::Left)?;
        let right = Vector::new(right, VectorKind::Right)?;
        validate_dimensions(&left, &right)?;
        let left = Norm::from_vector(&left, VectorKind::Left)?;
        let right = Norm::from_vector(&right, VectorKind::Right)?;
        Ok(Self { left, right })
    }

    /// Returns the stored norm for the left vector.
    #[must_use]
    pub fn left(&self) -> f32 {
        self.left.value()
    }

    /// Returns the stored norm for the right vector.
    #[must_use]
    pub fn right(&self) -> f32 {
        self.right.value()
    }

    /// Returns the validated norm for the left vector.
    #[must_use]
    pub fn left_norm(&self) -> Norm {
        self.left
    }

    /// Returns the validated norm for the right vector.
    #[must_use]
    pub fn right_norm(&self) -> Norm {
        self.right
    }
}

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

    let mut dot = 0.0f64;
    let mut left_squares = 0.0f64;
    let mut right_squares = 0.0f64;
    let mut left_has_magnitude = false;
    let mut right_has_magnitude = false;

    for (&l, &r) in left.iter().zip(right.iter()) {
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
            (norms.left_norm(), norms.right_norm())
        }
        None => (
            Norm::new(left_squares.sqrt() as f32, VectorKind::Left)?,
            Norm::new(right_squares.sqrt() as f32, VectorKind::Right)?,
        ),
    };

    let denominator = f64::from(*left_norm) * f64::from(*right_norm);
    let similarity = (dot / denominator) as f32;
    // Theoretical range is [-1, 1], but numerical noise can spill over.
    let similarity = similarity.clamp(-1.0, 1.0);

    Ok(Distance::from_raw(1.0 - similarity))
}

fn validate_dimensions(left: &Vector<'_>, right: &Vector<'_>) -> Result<()> {
    if left.dimension() != right.dimension() {
        return Err(DistanceError::DimensionMismatch {
            left: left.dimension(),
            right: right.dimension(),
        });
    }
    Ok(())
}
