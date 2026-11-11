//! CPU minimum spanning tree (MST) construction.
//!
//! This module provides a parallel Kruskal implementation intended for CPU
//! backends. The algorithm parallelizes the global edge sort via Rayon and
//! performs concurrent cycle checks using a striped-lock union-find.

mod error;

#[cfg(not(kani))]
mod union_find;

#[cfg(any(kani, test))]
mod kani_model;

use std::cmp::Ordering;

#[cfg(not(kani))]
use rayon::prelude::*;

use crate::{CandidateEdge, EdgeHarvest};

pub use self::error::{MstError, MstErrorCode};

#[cfg(not(kani))]
use self::union_find::ConcurrentUnionFind;

/// A single MST edge in canonical undirected form (`source <= target`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MstEdge {
    /// Smaller endpoint identifier in canonical order.
    source: usize,
    /// Larger endpoint identifier in canonical order.
    target: usize,
    /// Validated finite edge weight.
    weight: f32,
    /// Deterministic insertion-order tie breaker.
    sequence: u64,
}

impl MstEdge {
    /// Returns the smaller endpoint id.
    #[must_use]
    #[rustfmt::skip]
    pub const fn source(&self) -> usize { self.source }

    /// Returns the larger endpoint id.
    #[must_use]
    #[rustfmt::skip]
    pub const fn target(&self) -> usize { self.target }

    /// Returns the edge weight.
    #[must_use]
    #[rustfmt::skip]
    pub const fn weight(&self) -> f32 { self.weight }

    /// Returns the deterministic tie-break sequence associated with the edge.
    #[must_use]
    #[rustfmt::skip]
    pub const fn sequence(&self) -> u64 { self.sequence }
}

impl Eq for MstEdge {}

impl Ord for MstEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight
            .total_cmp(&other.weight)
            .then_with(|| self.source.cmp(&other.source))
            .then_with(|| self.target.cmp(&other.target))
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

impl PartialOrd for MstEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The output of a minimum spanning forest computation.
///
/// When the input graph is connected, the forest is a minimum spanning tree.
#[cfg(not(kani))]
#[derive(Clone, Debug, PartialEq)]
pub struct MinimumSpanningForest {
    /// Accepted minimum-weight edges in deterministic order.
    edges: Vec<MstEdge>,
    /// Number of connected components remaining after Kruskal processing.
    component_count: usize,
}

/// Bounded Kani representation of a minimum spanning forest.
///
/// The Kani harnesses exercise at most four nodes, so the forest has at most
/// three edges. Keeping that representation inline avoids modelling allocator
/// and panic paths that are unrelated to forest invariants.
#[cfg(kani)]
#[derive(Clone, Debug, PartialEq)]
pub struct MinimumSpanningForest {
    edges: [MstEdge; 3],
    edge_count: usize,
    component_count: usize,
}

impl MinimumSpanningForest {
    /// Returns the MST/forest edges.
    #[must_use]
    #[rustfmt::skip]
    #[cfg(not(kani))]
    pub fn edges(&self) -> &[MstEdge] { &self.edges }

    /// Returns the MST/forest edges in the Kani bounded model.
    #[must_use]
    #[rustfmt::skip]
    #[cfg(kani)]
    pub fn edges(&self) -> &[MstEdge] { &self.edges[..self.edge_count] }

    /// Returns the number of connected components in the resulting forest.
    #[must_use]
    #[rustfmt::skip]
    pub const fn component_count(&self) -> usize { self.component_count }

    /// Returns `true` when the forest spans a single connected component.
    #[must_use]
    pub const fn is_tree(&self) -> bool {
        self.component_count == 1
    }
}

/// Computes a minimum spanning forest using parallel Kruskal's algorithm.
///
/// The input edges are interpreted as undirected and are canonicalized to
/// `(min(u, v), max(u, v))`. Self-edges are ignored.
///
/// # Errors
///
/// Returns an error when:
/// - `node_count == 0`; this is decided before any edge is inspected
/// - an edge references a node id `>= node_count`
/// - an edge weight is non-finite
///
/// When several edges are invalid, the reported error is the one at the lowest
/// index in [`EdgeHarvest::as_slice`] order. Within one edge, the source bound
/// is checked before the target bound, and both before weight finiteness;
/// [`MstError::NonFiniteWeight`] reports the original `(source, target)`
/// orientation.
///
/// # Examples
///
/// ```
/// use chutoro_core::{CandidateEdge, EdgeHarvest, MstError, parallel_kruskal};
///
/// let harvest = EdgeHarvest::new(vec![
///     CandidateEdge::new(0, 1, 1.0, 0),
///     CandidateEdge::new(5, 0, 1.0, 1),
///     CandidateEdge::new(0, 7, 1.0, 2),
/// ]);
///
/// assert_eq!(
///     parallel_kruskal(3, &harvest),
///     Err(MstError::InvalidNodeId {
///         node: 5,
///         node_count: 3,
///     }),
/// );
/// ```
pub fn parallel_kruskal(
    node_count: usize,
    edges: &EdgeHarvest,
) -> Result<MinimumSpanningForest, MstError> {
    parallel_kruskal_from_edges(node_count, edges.as_slice())
}

/// Validate a candidate edge and convert it to canonical MST form.
const fn validate_and_canonicalize_edge(
    edge: &CandidateEdge,
    node_count: usize,
) -> Result<Option<MstEdge>, MstError> {
    let edge_source = edge.source();
    let edge_target = edge.target();

    if edge_source >= node_count {
        return Err(MstError::InvalidNodeId {
            node: edge_source,
            node_count,
        });
    }
    if edge_target >= node_count {
        return Err(MstError::InvalidNodeId {
            node: edge_target,
            node_count,
        });
    }

    let weight = edge.distance();
    if !weight.is_finite() {
        return Err(MstError::NonFiniteWeight {
            left: edge_source,
            right: edge_target,
        });
    }

    if edge_source == edge_target {
        return Ok(None);
    }

    let (canonical_source, canonical_target) = if edge_source <= edge_target {
        (edge_source, edge_target)
    } else {
        (edge_target, edge_source)
    };

    Ok(Some(MstEdge {
        source: canonical_source,
        target: canonical_target,
        weight,
        sequence: edge.sequence(),
    }))
}

/// Append non-cycling edges from one equal-weight group deterministically.
///
/// `group` arrives sorted by `(weight, source, target, sequence)`. Scanning it
/// sequentially and appending accepted edges straight to `forest_edges` keeps
/// that order deterministic and allocates nothing per weight group.
///
/// The forest is reused across groups, so a group's accepted edges are added to
/// whatever earlier groups already contributed. The bound that holds is on the
/// *increase*, not on the resulting total length.
///
/// # Examples
///
/// ```ignore
/// // An earlier equal-weight group already accepted the `(0, 1)` edge, so the
/// // forest is not empty when this group is processed.
/// let union_find = ConcurrentUnionFind::new(3);
/// union_find.try_union(first_edge.source(), first_edge.target())?;
/// let mut forest_edges = vec![first_edge];
///
/// // `(0, 2)` joins the last node; `(1, 2)` would close a cycle.
/// let joins_last_node = edge_between(0, 2);
/// let closes_cycle = edge_between(1, 2);
/// let group = [joins_last_node, closes_cycle];
/// let before = forest_edges.len();
///
/// extend_with_accepted_edges(&group, &union_find, &mut forest_edges)?;
///
/// // Only the first edge was appended, and the bound is on the increase
/// // rather than on the resulting total length.
/// assert_eq!(forest_edges.len(), before + 1);
/// assert_eq!(forest_edges.last(), Some(&joins_last_node));
/// assert!(forest_edges.len() - before <= group.len());
/// ```
#[cfg(not(kani))]
fn extend_with_accepted_edges(
    group: &[MstEdge],
    union_find: &ConcurrentUnionFind,
    forest_edges: &mut Vec<MstEdge>,
) -> Result<(), MstError> {
    for edge in group {
        if union_find.try_union(edge.source, edge.target)? {
            forest_edges.push(*edge);
        }
    }
    Ok(())
}

#[cfg(not(kani))]
/// Report whether a forest contains the edge count of a spanning tree.
fn is_mst_complete(
    node_count: usize,
    union_find: &ConcurrentUnionFind,
    forest_edges: &[MstEdge],
) -> bool {
    union_find.components() == 1 && forest_edges.len() == node_count.saturating_sub(1)
}

/// Validate, canonicalize, sort, and deduplicate candidate edges.
///
/// Validation runs in parallel but error selection is sequential: the results
/// are collected into a buffer in input order, then scanned from the start, so
/// the reported error is always the one belonging to the lowest input index.
/// "Input index" means the position in `edges`, which for the public entry
/// point is [`EdgeHarvest::as_slice`] order.
///
/// This is deliberately not a fallible parallel collection. Rayon documents
/// that a parallel `try_reduce` returns an unspecified error when several are
/// present, and that collecting into a `Result` is likewise nondeterministic
/// when more than one failure occurs. Both properties come from a shared abort
/// flag: the first task to fail publishes an error and other tasks may stop
/// early with a partial result, so a later reduction can surface whichever
/// error happened to win the race rather than the lowest-index one. The
/// ordered buffer makes the lowest-index rule true by construction instead of
/// by a post-hoc minimum reduction.
#[cfg(not(kani))]
fn prepare_edge_list(edges: &[CandidateEdge], node_count: usize) -> Result<Vec<MstEdge>, MstError> {
    let mut validated = Vec::with_capacity(edges.len());
    edges
        .par_iter()
        .map(|edge| validate_and_canonicalize_edge(edge, node_count))
        .collect_into_vec(&mut validated);

    let mut edge_list = Vec::with_capacity(edges.len());
    for result in validated {
        match result {
            Ok(Some(mst_edge)) => edge_list.push(mst_edge),
            Ok(None) => {}
            Err(error) => return Err(error),
        }
    }

    edge_list.par_sort_unstable();
    edge_list.dedup_by(|left, right| {
        matches!(
            left.weight.partial_cmp(&right.weight),
            Some(Ordering::Equal)
        ) && left.source == right.source
            && left.target == right.target
    });
    Ok(edge_list)
}

/// Run parallel Kruskal directly over a slice of candidate edges.
pub(crate) fn parallel_kruskal_from_edges(
    node_count: usize,
    edges: &[CandidateEdge],
) -> Result<MinimumSpanningForest, MstError> {
    #[cfg(kani)]
    {
        return kani_model::parallel_kruskal_from_edges_for_kani(node_count, edges);
    }

    #[cfg(not(kani))]
    {
        if node_count == 0 {
            return Err(MstError::EmptyGraph);
        }

        let edge_list = prepare_edge_list(edges, node_count)?;

        if edge_list.is_empty() {
            return Ok(MinimumSpanningForest {
                edges: Vec::new(),
                component_count: node_count,
            });
        }

        let union_find = ConcurrentUnionFind::new(node_count);
        let mut forest_edges = Vec::with_capacity(node_count.saturating_sub(1));

        for group in edge_list.chunk_by(|left, right| {
            matches!(
                left.weight.partial_cmp(&right.weight),
                Some(Ordering::Equal)
            )
        }) {
            extend_with_accepted_edges(group, &union_find, &mut forest_edges)?;

            if is_mst_complete(node_count, &union_find, &forest_edges) {
                break;
            }
        }

        forest_edges.sort_unstable();
        Ok(MinimumSpanningForest {
            edges: forest_edges,
            component_count: union_find.components(),
        })
    }
}

#[cfg(kani)]
mod kani_harness;

#[cfg(test)]
pub(crate) mod property;

#[cfg(test)]
mod tests;
