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
    /// Index of the neighbour within the [`crate::DataSource`].
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RankedNeighbour {
    pub(crate) neighbour: Neighbour,
    pub(crate) sequence: u64,
}

impl RankedNeighbour {
    pub(crate) fn new(id: usize, distance: f32, sequence: u64) -> Self {
        Self {
            neighbour: Neighbour { id, distance },
            sequence,
        }
    }

    pub(crate) fn into_neighbour(self) -> Neighbour {
        self.neighbour
    }
}

impl Eq for RankedNeighbour {}

impl Ord for RankedNeighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.neighbour
            .distance
            .partial_cmp(&other.neighbour.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.neighbour.id.cmp(&other.neighbour.id))
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

impl PartialOrd for RankedNeighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReverseNeighbour {
    pub(crate) inner: RankedNeighbour,
}

impl ReverseNeighbour {
    pub(crate) fn new(id: usize, distance: f32, sequence: u64) -> Self {
        Self {
            inner: RankedNeighbour::new(id, distance, sequence),
        }
    }

    pub(crate) fn from_ranked(neighbour: RankedNeighbour) -> Self {
        Self { inner: neighbour }
    }
}

impl Eq for ReverseNeighbour {}

impl Ord for ReverseNeighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .inner
            .neighbour
            .distance
            .partial_cmp(&self.inner.neighbour.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.inner.neighbour.id.cmp(&other.inner.neighbour.id))
            .then_with(|| self.inner.sequence.cmp(&other.inner.sequence))
    }
}

impl PartialEq for ReverseNeighbour {
    fn eq(&self, other: &Self) -> bool {
        self.inner.neighbour.distance == other.inner.neighbour.distance
            && self.inner.neighbour.id == other.inner.neighbour.id
            && self.inner.sequence == other.inner.sequence
    }
}

impl PartialOrd for ReverseNeighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
