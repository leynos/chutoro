//! Data source abstractions for the Chutoro core runtime.

use crate::error::DataSourceError;
use std::{fmt, sync::Arc};

/// Describes the distance metric exposed by a [`DataSource`].
///
/// The identifier must include all configuration that affects distance
/// semantics. For example, cosine distance with pre-computed norms should
/// expose a different descriptor to the raw cosine metric so that caches can
/// distinguish them.
///
/// # Examples
/// ```
/// use chutoro_core::MetricDescriptor;
///
/// let descriptor = MetricDescriptor::new("cosine:prenorm=true");
/// assert_eq!(descriptor.as_str(), "cosine:prenorm=true");
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MetricDescriptor(Arc<str>);

impl MetricDescriptor {
    /// Creates a descriptor from a string identifier.
    #[must_use]
    pub fn new(identifier: impl Into<Arc<str>>) -> Self {
        Self(identifier.into())
    }

    /// Returns the metric identifier as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Builds the default "unknown" descriptor.
    #[must_use]
    pub fn unknown() -> Self {
        Self::new("unknown")
    }
}

impl Default for MetricDescriptor {
    fn default() -> Self {
        Self::unknown()
    }
}

impl fmt::Display for MetricDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
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
///
/// let batched = src.batch_distances(0, &[1, 2])?;
/// assert_eq!(batched, [1.0, 3.0]);
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

    /// Returns the descriptor for the metric used by the data source.
    ///
    /// The default implementation returns [`MetricDescriptor::unknown`].
    /// Implementations should override this method to provide a stable,
    /// configuration-rich identifier so that caches can disambiguate entries
    /// for different metrics or pre-processing modes.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{DataSource, MetricDescriptor};
    ///
    /// struct Dummy;
    ///
    /// impl DataSource for Dummy {
    ///     fn len(&self) -> usize { 0 }
    ///     fn name(&self) -> &str { "dummy" }
    ///     fn distance(&self, _: usize, _: usize) -> Result<f32, chutoro_core::DataSourceError> {
    ///         Ok(0.0)
    ///     }
    ///     fn metric_descriptor(&self) -> MetricDescriptor {
    ///         MetricDescriptor::new("euclidean")
    ///     }
    /// }
    ///
    /// let src = Dummy;
    /// assert_eq!(src.metric_descriptor().as_str(), "euclidean");
    /// ```
    #[must_use]
    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::unknown()
    }

    /// Computes the distance between two items.
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError>;

    /// Computes the distances from `query` to every entry in `candidates`.
    ///
    /// Implementations can override this method to provide SIMD-optimised
    /// kernels. The default implementation calls [`Self::distance`] repeatedly and
    /// collects the results.
    ///
    /// # Errors
    /// Returns any [`DataSourceError`] surfaced by [`Self::distance`]. Implementations
    /// must return [`DataSourceError::OutOfBounds`] for invalid indices and must
    /// not yield non-finite distances; callers may validate and fail on NaNs.
    fn batch_distances(
        &self,
        query: usize,
        candidates: &[usize],
    ) -> Result<Vec<f32>, DataSourceError> {
        candidates
            .iter()
            .map(|&candidate| self.distance(query, candidate))
            .collect()
    }

    /// Computes several distances at once, storing results in `out`.
    ///
    /// The default implementation calls [`Self::distance`] for each pair.
    ///
    /// # Errors
    /// Returns `DataSourceError::OutputLengthMismatch` if `pairs.len() != out.len()`.
    ///
    /// If any pair fails, `out` is left unmodified.
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
        // Compute into a temp buffer to keep `out` unchanged on error.
        let mut tmp = vec![0.0_f32; pairs.len()];
        for (idx, (i, j)) in pairs.iter().enumerate() {
            tmp[idx] = self.distance(*i, *j)?;
        }
        out.copy_from_slice(&tmp);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::CountingSource;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn batch_distances_invokes_scalar_distance() {
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingSource::new(vec![0.0, 1.0, 3.0], Arc::clone(&calls));

        let distances = source
            .batch_distances(0, &[1, 2])
            .expect("batch distances should succeed");

        assert_eq!(distances, vec![1.0, 3.0]);
        assert_eq!(source.calls().load(Ordering::Relaxed), 2);
    }

    #[test]
    fn batch_distances_propagates_errors() {
        let calls = Arc::new(AtomicUsize::new(0));
        let source = CountingSource::new(vec![0.0, 1.0], calls);

        let err = source
            .batch_distances(0, &[1, 5])
            .expect_err("invalid candidate must fail");

        assert!(
            matches!(err, DataSourceError::OutOfBounds { index: 5 }),
            "expected OutOfBounds with index 5, got {err:?}",
        );
    }
}
