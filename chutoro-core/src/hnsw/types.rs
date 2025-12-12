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
/// Wraps a `Vec<CandidateEdge>` to provide a stable API that can evolve
/// (e.g., to carry metadata or change representation) without breaking
/// consumers.
///
/// Used to accumulate edges from parallel insertions via Rayon `map` → `reduce`
/// for subsequent MST construction.
///
/// # Examples
/// ```
/// use chutoro_core::{CandidateEdge, EdgeHarvest};
///
/// let edges = vec![
///     CandidateEdge::new(0, 1, 0.5, 1),
///     CandidateEdge::new(1, 2, 0.3, 2),
/// ];
/// let harvest = EdgeHarvest::new(edges);
/// assert_eq!(harvest.len(), 2);
/// assert!(!harvest.is_empty());
///
/// for edge in harvest.iter() {
///     assert!(edge.distance() > 0.0);
/// }
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EdgeHarvest(Vec<CandidateEdge>);

impl EdgeHarvest {
    /// Creates a new edge harvest from the given edges, applying deterministic ordering.
    ///
    /// Edges are sorted primarily by insertion sequence (for deterministic ordering
    /// across parallel insertions), then by the natural `Ord` implementation
    /// (distance, source, target, sequence).
    ///
    /// All constructors enforce this invariant to ensure consistent behaviour.
    #[must_use]
    pub fn new(edges: Vec<CandidateEdge>) -> Self {
        Self::from_unsorted(edges)
    }

    /// Creates an edge harvest from unsorted edges, applying deterministic ordering.
    ///
    /// Edges are sorted primarily by insertion sequence (for deterministic ordering
    /// across parallel insertions), then by the natural `Ord` implementation
    /// (distance, source, target, sequence).
    #[must_use]
    pub fn from_unsorted(mut edges: Vec<CandidateEdge>) -> Self {
        edges.sort_unstable_by(|a, b| a.sequence().cmp(&b.sequence()).then_with(|| a.cmp(b)));
        Self(edges)
    }

    /// Returns the number of harvested edges.
    #[must_use]
    #[rustfmt::skip]
    pub fn len(&self) -> usize { self.0.len() }

    /// Returns whether the harvest contains no edges.
    #[must_use]
    #[rustfmt::skip]
    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    /// Returns an iterator over the harvested edges.
    #[rustfmt::skip]
    pub fn iter(&self) -> impl Iterator<Item = &CandidateEdge> { self.0.iter() }

    /// Consumes the harvest and returns the underlying edges.
    #[must_use]
    #[rustfmt::skip]
    pub fn into_inner(self) -> Vec<CandidateEdge> { self.0 }

    /// Returns an iterator over overlapping windows of edges.
    ///
    /// Used for verifying sort order invariants.
    pub fn windows(&self, size: usize) -> impl Iterator<Item = &[CandidateEdge]> {
        self.0.windows(size)
    }
}

impl From<Vec<CandidateEdge>> for EdgeHarvest {
    fn from(edges: Vec<CandidateEdge>) -> Self {
        Self::from_unsorted(edges)
    }
}

impl IntoIterator for EdgeHarvest {
    type Item = CandidateEdge;
    type IntoIter = std::vec::IntoIter<CandidateEdge>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a EdgeHarvest {
    type Item = &'a CandidateEdge;
    type IntoIter = std::slice::Iter<'a, CandidateEdge>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
