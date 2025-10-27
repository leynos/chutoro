//! In-memory test support for property-based HNSW validation.
//!
//! Provides [`DenseVectorSource`] for wrapping generated vector datasets with
//! eager dimension validation alongside vector utilities used by the dataset
//! generators ([`euclidean_distance`], [`dot`], [`l2_norm`], [`unit_vector`]).

use std::sync::Arc;

use crate::{DataSource, DataSourceError};

/// Dense in-memory data source backed by generated vectors for property tests.
///
/// Wraps generated vectors in `Arc` so multiple strategies can clone the data
/// cheaply. Implements [`DataSource`] to expose Euclidean distance queries and
/// eagerly validates dimensions, ensuring fixtures fail fast when provided with
/// malformed vectors.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::support::DenseVectorSource;
/// use crate::DataSourceError;
///
/// let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let source = DenseVectorSource::new("test", vectors)?;
/// # Ok::<(), DataSourceError>(())
/// ```
#[derive(Clone, Debug)]
pub(super) struct DenseVectorSource {
    name: Arc<str>,
    vectors: Arc<[Vec<f32>]>,
}

impl DenseVectorSource {
    /// Constructs a new dense source, validating dimensions eagerly.
    ///
    /// Returns an error if the dataset is empty, contains zero-length vectors,
    /// or mixes dimensions.
    pub fn new(name: impl Into<String>, vectors: Vec<Vec<f32>>) -> Result<Self, DataSourceError> {
        if vectors.is_empty() {
            return Err(DataSourceError::EmptyData);
        }
        let dimension = vectors[0].len();
        if dimension == 0 {
            return Err(DataSourceError::ZeroDimension);
        }
        for vector in &vectors {
            if vector.len() != dimension {
                return Err(DataSourceError::DimensionMismatch {
                    left: dimension,
                    right: vector.len(),
                });
            }
        }
        let name = Arc::<str>::from(name.into());
        Ok(Self {
            name,
            vectors: Arc::from(vectors),
        })
    }
}

impl DataSource for DenseVectorSource {
    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_vec = self
            .vectors
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let right_vec = self
            .vectors
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        if left_vec.len() != right_vec.len() {
            return Err(DataSourceError::DimensionMismatch {
                left: left_vec.len(),
                right: right_vec.len(),
            });
        }
        Ok(euclidean_distance(left_vec, right_vec))
    }
}

/// Computes the Euclidean distance between two vectors.
///
/// Assumes `left` and `right` have equal length; behaviour is unspecified when
/// they differ because the internal iterator zips both slices.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::support::euclidean_distance;
///
/// let distance = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
/// assert_eq!(distance, 5.0);
/// ```
pub(super) fn euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(l, r)| {
            let diff = l - r;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

/// Computes the dot product of two vectors.
///
/// Assumes `left` and `right` have equal length.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::support::dot;
///
/// let product = dot(&[1.0, 2.0], &[3.0, 4.0]);
/// assert_eq!(product, 11.0);
/// ```
pub(super) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

/// Computes the L2 (Euclidean) norm of a vector.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::support::l2_norm;
///
/// let norm = l2_norm(&[3.0, 4.0]);
/// assert_eq!(norm, 5.0);
/// ```
pub(super) fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Creates a unit vector with 1.0 at the specified axis.
///
/// Returns a vector of length `dimension` with all zeros except position
/// `axis`, which is set to 1.0. If `axis >= dimension`, returns an all-zero
/// vector.
///
/// # Examples
///
/// ```ignore
/// use crate::hnsw::tests::property::support::unit_vector;
///
/// let vector = unit_vector(3, 1);
/// assert_eq!(vector, vec![0.0, 1.0, 0.0]);
/// ```
pub(super) fn unit_vector(dimension: usize, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    if axis < dimension {
        vector[axis] = 1.0;
    }
    vector
}
