//! Chutoro core library.

use thiserror::Error;

/// An error produced by [`DataSource`] operations.
#[derive(Debug, Error)]
pub enum DataSourceError {
    /// Requested index was outside the source's bounds.
    #[error("index {index} is out of bounds")]
    OutOfBounds { index: usize },
    /// Provided output buffer length did not match number of pairs.
    #[error("output buffer has length {out} but {expected} pairs were given")]
    OutputLengthMismatch { out: usize, expected: usize },
}

/// Abstraction over a collection of items that can yield pairwise distances.
///
/// # Examples
/// ```
/// use chutoro_core::{DataSource, DataSourceError};
///
/// struct Dummy(Vec<f32>);
///
/// impl DataSource for Dummy {
///     fn len(&self) -> usize { self.0.len() }
///     fn name(&self) -> &str { "dummy" }
///     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
///         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
///         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
///         Ok((a - b).abs())
///     }
/// }
///
/// let src = Dummy(vec![1.0, 2.0, 4.0]);
/// assert_eq!(src.len(), 3);
/// assert_eq!(src.name(), "dummy");
/// assert_eq!(src.distance(0, 2)?, 3.0);
///
/// let pairs = vec![(0, 1), (1, 2)];
/// let mut out = vec![0.0; 2];
/// src.distance_batch(&pairs, &mut out)?;
/// assert_eq!(out, [1.0, 2.0]);
/// # Ok::<(), DataSourceError>(())
/// ```
pub trait DataSource {
    /// Returns number of items in the source.
    fn len(&self) -> usize;

    /// Returns whether the source contains no items.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{DataSource, DataSourceError};
    /// struct Empty;
    /// impl DataSource for Empty {
    ///     fn len(&self) -> usize { 0 }
    ///     fn name(&self) -> &str { "empty" }
    ///     fn distance(&self, _: usize, _: usize) -> Result<f32, DataSourceError> { Ok(0.0) }
    /// }
    /// let src = Empty;
    /// assert!(src.is_empty());
    /// ```
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a human-readable name.
    fn name(&self) -> &str;

    /// Computes the distance between two items.
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError>;

    /// Computes several distances at once, storing results in `out`.
    ///
    /// The default implementation calls [`distance`] for each pair.
    ///
    /// # Errors
    /// Returns `DataSourceError::OutputLengthMismatch` if `pairs.len() != out.len()`.
    fn distance_batch(
        &self,
        pairs: &[(usize, usize)],
        out: &mut [f32],
    ) -> Result<(), DataSourceError> {
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }
        for (idx, (i, j)) in pairs.iter().enumerate() {
            out[idx] = self.distance(*i, *j)?;
        }
        Ok(())
    }
}
