//! Types for HNSW graph operations (entry points, plans, and neighbour
//! ordering semantics). Distances are finite `f32` values enforced by the
//! validation layer.

use std::cmp::Ordering;

/// Entry point into the hierarchical graph used when searching.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EntryPoint {
    pub(crate) node: usize,
    pub(crate) level: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct InsertionPlan {
    pub(crate) layers: Vec<LayerPlan>,
}

#[derive(Clone, Debug)]
pub(crate) struct LayerPlan {
    pub(crate) level: usize,
    pub(crate) neighbours: Vec<Neighbour>,
}

/// Neighbour discovered during a search, including its distance from the query.
///
/// Distances must be finite (`f32::is_finite()`); non-finite values are invalid
/// and rejected by validation in the search pipeline.
///
/// # Examples
/// ```
/// use chutoro_core::Neighbour;
///
/// let neighbour = Neighbour { id: 3, distance: 0.42 };
/// assert_eq!(neighbour.id, 3);
/// assert!(neighbour.distance < 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbour {
    /// Index of the neighbour within the [`DataSource`].
    pub id: usize,
    /// Distance between the query item and [`Neighbour::id`].
    pub distance: f32,
}

impl Eq for Neighbour {}

impl Ord for Neighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReverseNeighbour {
    pub(crate) inner: Neighbour,
}

impl ReverseNeighbour {
    pub(crate) fn new(id: usize, distance: f32) -> Self {
        Self {
            inner: Neighbour { id, distance },
        }
    }
}

impl Eq for ReverseNeighbour {}

impl Ord for ReverseNeighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .inner
            .distance
            .partial_cmp(&self.inner.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.inner.id.cmp(&other.inner.id))
    }
}

impl PartialEq for ReverseNeighbour {
    fn eq(&self, other: &Self) -> bool {
        self.inner.distance == other.inner.distance && self.inner.id == other.inner.id
    }
}

impl PartialOrd for ReverseNeighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
