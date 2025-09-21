use std::collections::HashSet;

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

impl ClusteringResult {
    /// Builds a result from explicit cluster assignments.
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
        let cluster_count = assignments
            .iter()
            .map(|id| id.get())
            .collect::<HashSet<_>>()
            .len();
        Self {
            assignments,
            cluster_count,
        }
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
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying numeric identifier.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::ClusterId;
    ///
    /// let id = ClusterId::new(7);
    /// assert_eq!(id.get(), 7);
    /// ```
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}
