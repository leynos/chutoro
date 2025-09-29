//! Dense matrix data source implementations shared across ingestion paths.
use chutoro_core::{DataSource, DataSourceError};

/// In-memory dense vector data source.
pub struct DenseSource {
    data: Vec<Vec<f32>>,
    name: String,
}

impl DenseSource {
    #[cfg(test)]
    pub(crate) fn from_parts(name: impl Into<String>, data: Vec<Vec<f32>>) -> Self {
        Self {
            data,
            name: name.into(),
        }
    }

    /// Creates a new dense source.
    ///
    /// # Panics
    /// Panics if row lengths differ; use [`try_new`] for fallible construction.
    ///
    /// # Examples
    /// ```
    /// use chutoro_providers_dense::DenseSource;
    /// let ds = DenseSource::new("demo", vec![vec![0.0], vec![1.0]]);
    /// assert_eq!(ds.len(), 2);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(name: impl Into<String>, data: Vec<Vec<f32>>) -> Self {
        #[expect(
            clippy::expect_used,
            reason = "constructor panics on inconsistent row lengths"
        )]
        Self::try_new(name, data).expect("rows must have equal length")
    }

    /// Creates a dense source after validating uniform dimensions.
    ///
    /// # Errors
    /// Returns `DataSourceError::DimensionMismatch` if row lengths differ.
    /// Returns `DataSourceError::EmptyData` if `data` is empty.
    ///
    /// # Examples
    /// ```
    /// use chutoro_providers_dense::DenseSource;
    /// use chutoro_core::DataSourceError;
    /// let err = DenseSource::try_new("demo", vec![vec![0.0], vec![1.0, 2.0]]);
    /// assert!(matches!(err, Err(DataSourceError::DimensionMismatch { .. })));
    /// let err_empty = DenseSource::try_new("demo", vec![]);
    /// assert!(matches!(err_empty, Err(DataSourceError::EmptyData)));
    /// ```
    pub fn try_new(name: impl Into<String>, data: Vec<Vec<f32>>) -> Result<Self, DataSourceError> {
        if data.is_empty() {
            return Err(DataSourceError::EmptyData);
        }
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
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>();
        Ok(sum.sqrt())
    }
}
