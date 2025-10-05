//! Text provider for line-based UTF-8 sources implementing [`DataSource`].
use std::io::BufRead;

use chutoro_core::{DataSource, DataSourceError};
use strsim::levenshtein;
use thiserror::Error;

/// Errors produced when constructing a [`TextProvider`].
#[derive(Debug, Error)]
pub enum TextProviderError {
    /// The reader yielded no lines.
    #[error("text source is empty")]
    EmptyInput,
    /// Reading from the input failed.
    #[error("failed to read text source: {0}")]
    Io(#[from] std::io::Error),
}

/// UTF-8 text provider that reports Levenshtein distances between lines.
#[derive(Debug)]
pub struct TextProvider {
    data: Vec<String>,
    name: String,
}

impl TextProvider {
    /// Creates a provider from a collection of UTF-8 lines.
    ///
    /// # Errors
    /// Returns [`TextProviderError::EmptyInput`] when `lines` is empty.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::DataSource;
    /// use chutoro_providers_text::TextProvider;
    ///
    /// let provider = TextProvider::new("demo", vec!["kitten".into(), "sitting".into()])
    ///     .expect("provider must build");
    /// assert_eq!(provider.len(), 2);
    /// let distance = provider
    ///     .distance(0, 1)
    ///     .expect("distance calculation must succeed");
    /// assert_eq!(distance, 3.0);
    /// ```
    pub fn new(name: impl Into<String>, lines: Vec<String>) -> Result<Self, TextProviderError> {
        if lines.is_empty() {
            return Err(TextProviderError::EmptyInput);
        }
        Ok(Self {
            data: lines,
            name: name.into(),
        })
    }

    /// Creates a provider by reading one UTF-8 string per line from `reader`.
    ///
    /// # Errors
    /// Returns [`TextProviderError::EmptyInput`] if `reader` produced no lines and
    /// [`TextProviderError::Io`] if reading fails.
    ///
    /// # Examples
    /// ```
    /// use std::io::Cursor;
    ///
    /// use chutoro_core::DataSource;
    /// use chutoro_providers_text::TextProvider;
    ///
    /// let cursor = Cursor::new("alpha\\nbeta\\n");
    /// let provider = TextProvider::try_from_reader("demo", cursor)
    ///     .expect("provider must build");
    /// assert_eq!(provider.len(), 2);
    /// assert_eq!(provider.distance(0, 1).unwrap(), 4.0);
    /// ```
    pub fn try_from_reader(
        name: impl Into<String>,
        reader: impl BufRead,
    ) -> Result<Self, TextProviderError> {
        let mut lines = Vec::new();
        Self::read_lines(reader, &mut lines)?;
        Self::new(name, lines)
    }

    /// Returns the stored UTF-8 lines.
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.data
    }

    fn read_lines(
        mut reader: impl BufRead,
        lines: &mut Vec<String>,
    ) -> Result<(), TextProviderError> {
        let mut buffer = String::new();
        loop {
            buffer.clear();
            let bytes_read = reader.read_line(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            let line = buffer.trim_end_matches(['\r', '\n']).to_owned();
            lines.push(line);
        }
        if lines.is_empty() {
            return Err(TextProviderError::EmptyInput);
        }
        Ok(())
    }
}

impl DataSource for TextProvider {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        &self.name
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "Distances are exposed as f32 to match the DataSource API."
    )]
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let left = self
            .data
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self
            .data
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        let distance = levenshtein(left, right);
        Ok(distance as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, BufRead, Cursor, Read};

    struct FailingReader;

    impl Read for FailingReader {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("boom"))
        }
    }

    impl BufRead for FailingReader {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Err(io::Error::other("boom"))
        }

        fn consume(&mut self, _amt: usize) {}

        fn read_line(&mut self, _buf: &mut String) -> io::Result<usize> {
            Err(io::Error::other("boom"))
        }
    }

    #[test]
    fn read_lines_populates_collection() {
        let mut lines = Vec::new();
        TextProvider::read_lines(Cursor::new("alpha\nbeta\n"), &mut lines)
            .expect("reading must succeed");
        assert_eq!(lines, ["alpha", "beta"]);
    }

    #[test]
    fn read_lines_rejects_empty_input() {
        let mut lines = Vec::new();
        let err = TextProvider::read_lines(Cursor::new(""), &mut lines)
            .expect_err("empty input must fail");
        assert!(matches!(err, TextProviderError::EmptyInput));
    }

    #[test]
    fn read_lines_propagates_io_errors() {
        let mut lines = Vec::new();
        let err = TextProvider::read_lines(FailingReader, &mut lines)
            .expect_err("I/O failures must propagate");
        assert!(matches!(err, TextProviderError::Io(_)));
    }
}
