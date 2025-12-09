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
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Internal wrapper retaining deterministic ordering metadata for neighbour
/// comparisons.
///
/// # Examples
/// ```rust,ignore
/// use crate::hnsw::types::RankedNeighbour;
///
/// let ranked = RankedNeighbour::new(4, 0.5, 7);
/// assert_eq!(ranked.into_neighbour().id, 4);
/// ```
#[derive(Clone, Copy, Debug)]
pub(crate) struct RankedNeighbour {
    inner: Neighbour,
    sequence: u64,
}

impl RankedNeighbour {
    pub(crate) fn new(id: usize, distance: f32, sequence: u64) -> Self {
        Self {
            inner: Neighbour { id, distance },
            sequence,
        }
    }

    pub(crate) fn into_neighbour(self) -> Neighbour {
        self.inner
    }

    pub(crate) fn compare(&self, other: &Self) -> Ordering {
        self.inner
            .cmp(&other.inner)
            .then(self.sequence.cmp(&other.sequence))
    }
}

impl Eq for RankedNeighbour {}

impl PartialEq for RankedNeighbour {
    fn eq(&self, other: &Self) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

impl Ord for RankedNeighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(other)
    }
}

#[expect(
    clippy::non_canonical_partial_ord_impl,
    reason = "Reviewer requested direct delegation to compare()"
)]
impl PartialOrd for RankedNeighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

/// Edge discovered during HNSW insertion for MST construction.
///
/// Represents a candidate edge `(source, target, distance)` discovered when
/// inserting a node into the HNSW graph. These edges are collected during
/// the build phase and used for subsequent MST construction in the FISHDBC
/// pipeline.
///
/// The `sequence` field provides deterministic tie-breaking when edges have
/// equal distances, ensuring reproducible results under fixed RNG seeds.
///
/// # Examples
/// ```
/// use chutoro_core::CandidateEdge;
///
/// let edge = CandidateEdge::new(0, 1, 0.5, 42);
/// assert_eq!(edge.source(), 0);
/// assert_eq!(edge.target(), 1);
/// assert!((edge.distance() - 0.5).abs() < f32::EPSILON);
///
/// // Canonicalise ensures source <= target for undirected graphs.
/// let reversed = CandidateEdge::new(5, 2, 0.3, 10);
/// let canonical = reversed.canonicalise();
/// assert_eq!(canonical.source(), 2);
/// assert_eq!(canonical.target(), 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateEdge {
    source: usize,
    target: usize,
    distance: f32,
    sequence: u64,
}

impl CandidateEdge {
    /// Creates a new candidate edge.
    #[must_use]
    pub fn new(source: usize, target: usize, distance: f32, sequence: u64) -> Self {
        Self {
            source,
            target,
            distance,
            sequence,
        }
    }

    /// Returns the source node identifier.
    #[must_use]
    #[rustfmt::skip]
    pub fn source(&self) -> usize { self.source }

    /// Returns the target node identifier.
    #[must_use]
    #[rustfmt::skip]
    pub fn target(&self) -> usize { self.target }

    /// Returns the distance (weight) between source and target.
    #[must_use]
    #[rustfmt::skip]
    pub fn distance(&self) -> f32 { self.distance }

    /// Returns the insertion sequence for deterministic ordering.
    #[must_use]
    #[rustfmt::skip]
    pub fn sequence(&self) -> u64 { self.sequence }

    /// Returns the edge with `source <= target` for canonical representation.
    ///
    /// Useful for undirected MST construction where edge direction is
    /// irrelevant.
    #[must_use]
    pub fn canonicalise(self) -> Self {
        if self.source <= self.target {
            self
        } else {
            Self {
                source: self.target,
                target: self.source,
                ..self
            }
        }
    }
}

impl Eq for CandidateEdge {}

impl Ord for CandidateEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.source.cmp(&other.source))
            .then_with(|| self.target.cmp(&other.target))
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

impl PartialOrd for CandidateEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Collection of candidate edges discovered during HNSW construction.
///
/// Used to accumulate edges from parallel insertions via Rayon `map` → `reduce`
/// for subsequent MST construction.
pub type EdgeHarvest = Vec<CandidateEdge>;
