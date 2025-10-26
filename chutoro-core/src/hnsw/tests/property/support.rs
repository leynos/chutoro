use std::sync::Arc;

use crate::{DataSource, DataSourceError};

/// Dense in-memory data source backed by generated vectors.
#[derive(Clone, Debug)]
pub(super) struct DenseVectorSource {
    name: Arc<str>,
    vectors: Arc<[Vec<f32>]>,
}

impl DenseVectorSource {
    /// Constructs a new dense source, validating dimensions eagerly.
    pub fn new(name: impl Into<String>, vectors: Vec<Vec<f32>>) -> Result<Self, DataSourceError> {
        if vectors.is_empty() {
            return Err(DataSourceError::EmptyData);
        }
        let dimension = vectors.first().ok_or(DataSourceError::EmptyData)?.len();
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

pub(super) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(l, r)| l * r).sum()
}

pub(super) fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

pub(super) fn unit_vector(dimension: usize, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    if axis < dimension {
        vector[axis] = 1.0;
    }
    vector
}
