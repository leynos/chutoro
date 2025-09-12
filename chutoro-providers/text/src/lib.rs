#![cfg_attr(not(any(test, doctest)), forbid(clippy::expect_used))]
use chutoro_core::{DataSource, DataSourceError};

/// UTF-8 text line data source.
pub struct TextSource {
    data: Vec<String>,
    name: String,
}

impl TextSource {
    /// Creates a new text source.
    ///
    /// # Examples
    /// ```
    /// use chutoro_providers_text::TextSource;
    /// let ds = TextSource::new("demo", vec!["a".into(), "bb".into()]);
    /// assert_eq!(ds.len(), 2);
    /// ```
    #[must_use]
    pub fn new(name: impl Into<String>, data: Vec<String>) -> Self {
        Self {
            data,
            name: name.into(),
        }
    }
}

impl DataSource for TextSource {
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
        Ok((a.chars().count() as i32 - b.chars().count() as i32).abs() as f32)
    }
}
