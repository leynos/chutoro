//! Dense provider for in-memory f32 vectors implementing DataSource.
#![cfg_attr(not(any(test, doctest)), forbid(clippy::expect_used))]
use chutoro_core::{DataSource, DataSourceError};

/// In-memory dense vector data source.
pub struct DenseSource {
    data: Vec<Vec<f32>>,
    name: String,
}

impl DenseSource {
    /// Creates a new dense source.
    ///
    /// # Examples
    /// ```
    /// use chutoro_providers_dense::DenseSource;
    /// let ds = DenseSource::new("demo", vec![vec![0.0], vec![1.0]]);
    /// assert_eq!(ds.len(), 2);
    /// ```
    #[must_use]
    pub fn new(name: impl Into<String>, data: Vec<Vec<f32>>) -> Self {
        Self::try_new(name, data).unwrap_or_else(|err| panic!("{err}"))
    }

    pub fn try_new(name: impl Into<String>, data: Vec<Vec<f32>>) -> Result<Self, DataSourceError> {
        if let Some((first, rest)) = data.split_first() {
            let dim = first.len();
            for row in rest {
                if row.len() != dim {
                    return Err(DataSourceError::DimensionMismatch {
                        left: dim,
                        right: row.len(),
                    });
                }
            }
        }
        Ok(Self {
            data,
            name: name.into(),
        })
    }
}

impl DataSource for DenseSource {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn name(&self) -> &str {
        &self.name
    }
    #[expect(clippy::float_arithmetic, reason = "vector arithmetic")]
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self
            .data
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .data
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        if a.len() != b.len() {
            return Err(DataSourceError::DimensionMismatch {
                left: a.len(),
                right: b.len(),
            });
        }
        let sum = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>();
        Ok(sum.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn distance_dimension_mismatch() {
        let ds = DenseSource {
            data: vec![vec![0.0], vec![1.0, 2.0]],
            name: "d".into(),
        };
        let err = ds.distance(0, 1).unwrap_err();
        assert!(matches!(err, DataSourceError::DimensionMismatch { .. }));
    }

    #[rstest]
    fn try_new_rejects_mismatched_rows() {
        let err = DenseSource::try_new("d", vec![vec![0.0], vec![1.0, 2.0]]);
        assert!(matches!(
            err,
            Err(DataSourceError::DimensionMismatch { .. })
        ));
    }
}
