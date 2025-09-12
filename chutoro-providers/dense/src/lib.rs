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
        Self {
            data,
            name: name.into(),
        }
    }
}

impl DataSource for DenseSource {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self
            .data
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .data
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        let sum = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>();
        Ok(sum.sqrt())
    }
}
