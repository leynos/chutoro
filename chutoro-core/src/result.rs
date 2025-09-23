//! Result types for clustering operations.
//!
//! Provides structures to represent clustering results including cluster
//! assignments and validation of cluster identifier constraints.

use std::collections::HashSet;
use thiserror::Error;

/// Represents the output of a [`Chutoro::run`] invocation.
///
/// # Examples
/// ```
/// use chutoro_core::{ClusteringResult, ClusterId};
///
/// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0), ClusterId::new(1)]);
/// assert_eq!(result.assignments().len(), 2);
/// assert_eq!(result.cluster_count(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusteringResult {
    assignments: Vec<ClusterId>,
    cluster_count: usize,
}

/// Error returned when cluster identifiers are not contiguous starting at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NonContiguousClusterIds {
    /// The assignments do not include cluster `0`.
    #[error("cluster identifiers must include 0")]
    MissingZero,
    /// The assignments skip identifiers.
    #[error("cluster identifiers must be contiguous without gaps")]
    Gap,
    /// The assignments contain duplicates that mask missing identifiers.
    #[error("cluster identifiers must not repeat identifiers")]
    Duplicate,
    /// The assignments require identifiers beyond the host pointer width.
    #[error("cluster identifiers exceed or reach the host pointer-width limit")]
    Overflow,
}

impl ClusteringResult {
    /// Builds a result from explicit cluster assignments.
    ///
    /// Cluster identifiers must start at zero and be contiguous. Use
    /// [`Self::try_from_assignments`] to handle arbitrary identifiers.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ClusteringResult, ClusterId};
    ///
    /// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0)]);
    /// assert_eq!(result.cluster_count(), 1);
    /// ```
    #[must_use]
    pub fn from_assignments(assignments: Vec<ClusterId>) -> Self {
        Self::try_from_assignments(assignments)
            .expect("cluster identifiers must start at zero and be contiguous")
    }

    /// Attempts to build a result from cluster assignments.
    ///
    /// The assignments must be contiguous starting at zero. This helper allows
    /// callers to surface meaningful errors instead of panicking when the
    /// invariant is violated.
    ///
    /// An empty `assignments` vector is accepted and yields `cluster_count == 0`.
    ///
    /// # Errors
    /// Returns [`NonContiguousClusterIds::MissingZero`] when the assignments omit
    /// cluster `0`, [`NonContiguousClusterIds::Gap`] when identifiers skip values,
    /// [`NonContiguousClusterIds::Duplicate`] when duplicates hide missing identifiers,
    /// and [`NonContiguousClusterIds::Overflow`] when identifiers exceed the host
    /// pointer width.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ClusteringResult, ClusterId};
    ///
    /// let result = ClusteringResult::try_from_assignments(vec![ClusterId::new(0)])
    ///     .expect("assignments are contiguous");
    /// assert_eq!(result.cluster_count(), 1);
    /// ```
    pub fn try_from_assignments(
        assignments: Vec<ClusterId>,
    ) -> Result<Self, NonContiguousClusterIds> {
        if assignments.is_empty() {
            return Ok(Self {
                assignments,
                cluster_count: 0,
            });
        }

        let mut seen = HashSet::new();
        let mut max_id = 0u64;
        let mut has_duplicate = false;

        for id in &assignments {
            let value = id.get();
            if value >= usize::MAX as u64 {
                return Err(NonContiguousClusterIds::Overflow);
            }
            if !seen.insert(value) {
                has_duplicate = true;
            }
            max_id = max_id.max(value);
        }

        if !seen.contains(&0) {
            return Err(NonContiguousClusterIds::MissingZero);
        }

        let expected = max_id
            .checked_add(1)
            .ok_or(NonContiguousClusterIds::Overflow)?;

        if expected > usize::MAX as u64 {
            return Err(NonContiguousClusterIds::Overflow);
        }

        let expected = expected as usize;

        if seen.len() != expected {
            return Err(if has_duplicate {
                NonContiguousClusterIds::Duplicate
            } else {
                NonContiguousClusterIds::Gap
            });
        }

        Ok(Self {
            assignments,
            cluster_count: seen.len(),
        })
    }

    /// Returns the assignments in insertion order.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ClusteringResult, ClusterId};
    ///
    /// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0)]);
    /// assert_eq!(result.assignments()[0].get(), 0);
    /// ```
    #[must_use]
    pub fn assignments(&self) -> &[ClusterId] {
        &self.assignments
    }

    /// Counts how many distinct clusters exist within the assignments.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::{ClusteringResult, ClusterId};
    ///
    /// let result = ClusteringResult::from_assignments(vec![ClusterId::new(0), ClusterId::new(0)]);
    /// assert_eq!(result.cluster_count(), 1);
    /// ```
    #[must_use]
    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }
}

/// Identifier assigned to a cluster.
///
/// # Examples
/// ```
/// use chutoro_core::ClusterId;
///
/// let id = ClusterId::new(4);
/// assert_eq!(id.get(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClusterId(u64);

impl ClusterId {
    /// Creates a new cluster identifier.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ClusterId;
    ///
    /// let id = ClusterId::new(2);
    /// assert_eq!(id.get(), 2);
    /// ```
    #[rustfmt::skip]
    #[must_use]
    pub fn new(id: u64) -> Self { Self(id) }

    /// Returns the underlying numeric identifier.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ClusterId;
    ///
    /// let id = ClusterId::new(7);
    /// assert_eq!(id.get(), 7);
    /// ```
    #[rustfmt::skip]
    #[must_use]
    pub fn get(self) -> u64 { self.0 }
}
