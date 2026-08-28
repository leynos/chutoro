//! Domain primitives shared by the distance routines.

use core::{
    fmt,
    ops::{AddAssign, Deref, Mul},
};

use num_traits::cast;
use thiserror::Error;

use super::helpers::validate_dimensions;

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
    DimensionMismatch {
        /// Dimensionality of the left-hand vector.
        left: usize,
        /// Dimensionality of the right-hand vector.
        right: usize,
    },
    /// Encountered a non-finite value in one of the vectors.
    #[error("{which} vector contains a non-finite value at index {index}: {value}")]
    NonFinite {
        /// Identifies whether the left or right vector contained the value.
        which: VectorKind,
        /// Offset within the offending vector.
        index: usize,
        /// The non-finite value that triggered validation failure.
        value: f32,
    },
    /// Cosine distance is undefined for zero-magnitude vectors.
    #[error("{which} vector has zero magnitude")]
    ZeroMagnitude {
        /// Identifies the vector whose magnitude collapsed to zero.
        which: VectorKind,
    },
    /// Provided norms must be finite.
    #[error("provided {which} norm must be finite (got {value})")]
    InvalidNorm {
        /// Identifies the vector whose norm was invalid.
        which: VectorKind,
        /// The invalid norm value supplied by the caller.
        value: f32,
    },
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
    pub const fn dimension(&self) -> usize {
        self.0.len()
    }
}

impl AsRef<[f32]> for Vector<'_> {
    fn as_ref(&self) -> &[f32] {
        self.0
    }
}

impl Deref for Vector<'_> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

/// Narrows an internal calculation to the public distance scalar type.
///
/// `num_traits` preserves Rust's floating-point narrowing behaviour, including
/// overflow to infinity, while making the deliberate conversion boundary
/// explicit.
pub(crate) fn narrow_to_f32(value: f64) -> f32 {
    cast(value).unwrap_or(f32::NAN)
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
            sum.add_assign(f64::from(*value).mul(f64::from(*value)));
        }

        Self::from_squared_sum(sum, which)
    }

    /// Validate a squared norm before converting it to a finite magnitude.
    pub(crate) fn validate_squared_sum(sum: f64, which: VectorKind) -> Result<()> {
        if !sum.is_finite() {
            return Err(DistanceError::InvalidNorm {
                which,
                value: f32::INFINITY,
            });
        }

        if sum == 0.0 {
            return Err(DistanceError::ZeroMagnitude { which });
        }

        Ok(())
    }

    /// Construct a norm from a validated squared sum.
    pub(crate) fn from_squared_sum(sum: f64, which: VectorKind) -> Result<Self> {
        Self::validate_squared_sum(sum, which)?;
        Self::new(narrow_to_f32(sum.sqrt()), which)
    }

    /// Returns the validated norm value.
    #[must_use]
    pub const fn value(&self) -> f32 {
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
    /// Construct a distance from a value already validated by its caller.
    pub(crate) const fn from_raw(value: f32) -> Self {
        Self(value)
    }

    /// Returns the raw distance value.
    #[must_use]
    pub const fn value(&self) -> f32 {
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
    /// Norm for the left vector.
    left: Norm,
    /// Norm for the right vector.
    right: Norm,
}

impl CosineNorms {
    /// Builds [`CosineNorms`] from explicit norm values.
    ///
    /// # Errors
    ///
    /// Returns [`DistanceError::InvalidNorm`] when a norm is non-finite or
    /// negative and [`DistanceError::ZeroMagnitude`] when a norm is zero.
    pub fn new(left_value: f32, right_value: f32) -> Result<Self> {
        let left_norm = Norm::new(left_value, VectorKind::Left)?;
        let right_norm = Norm::new(right_value, VectorKind::Right)?;
        Ok(Self {
            left: left_norm,
            right: right_norm,
        })
    }

    /// Computes norms from the provided vectors.
    ///
    /// # Errors
    ///
    /// Propagates validation errors surfaced by
    /// [`crate::distance::cosine_distance`].
    pub fn from_vectors(left_values: &[f32], right_values: &[f32]) -> Result<Self> {
        let left_vector = Vector::new(left_values, VectorKind::Left)?;
        let right_vector = Vector::new(right_values, VectorKind::Right)?;
        validate_dimensions(&left_vector, &right_vector)?;
        let left_norm = Norm::from_vector(&left_vector, VectorKind::Left)?;
        let right_norm = Norm::from_vector(&right_vector, VectorKind::Right)?;
        Ok(Self {
            left: left_norm,
            right: right_norm,
        })
    }

    /// Returns the stored norm for the left vector.
    #[must_use]
    pub const fn left(&self) -> f32 {
        self.left.value()
    }

    /// Returns the stored norm for the right vector.
    #[must_use]
    pub const fn right(&self) -> f32 {
        self.right.value()
    }

    /// Returns the validated norm for the left vector.
    #[must_use]
    pub const fn left_norm(&self) -> Norm {
        self.left
    }

    /// Returns the validated norm for the right vector.
    #[must_use]
    pub const fn right_norm(&self) -> Norm {
        self.right
    }
}
