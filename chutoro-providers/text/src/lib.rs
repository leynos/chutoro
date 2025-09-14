//! Text provider for line-based UTF-8 sources implementing DataSource.
use chutoro_core::{DataSource, DataSourceError};

/// UTF-8 text line data source.
pub struct TextSource {
    data: Vec<String>,
    lengths: Vec<usize>, // cached character counts
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
        let lengths = data.iter().map(|s| s.chars().count()).collect();
        // Cache character counts to avoid repeated iteration in `distance`.
        Self {
            data,
            lengths,
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
        let da = *self
            .lengths
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let db = *self
            .lengths
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok(da.abs_diff(db) as f32)
    }
}
