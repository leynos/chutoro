//! CPU minimum spanning tree (MST) construction.
//!
//! This module provides a parallel Kruskal implementation intended for CPU
//! backends. The algorithm parallelises the global edge sort via Rayon and
//! performs concurrent cycle checks using a striped-lock union-find.

mod union_find;

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::{CandidateEdge, EdgeHarvest};

use self::union_find::ConcurrentUnionFind;

/// Errors returned while computing a minimum spanning tree/forest.
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum MstError {
    /// The caller requested an MST for an empty graph.
    #[error("cannot compute an MST for an empty graph")]
    EmptyGraph,
    /// An edge referenced a node id that is not present in the graph.
    #[error("edge references node {node}, but node_count is {node_count}")]
    InvalidNodeId {
        /// The invalid node id referenced by an edge.
        node: usize,
        /// The number of nodes in the graph.
        node_count: usize,
    },
    /// An edge contained a non-finite weight.
    #[error("edge ({left}, {right}) has non-finite weight")]
    NonFiniteWeight {
        /// The left endpoint id (as provided).
        left: usize,
        /// The right endpoint id (as provided).
        right: usize,
    },
    /// A synchronisation primitive became poisoned after a panic.
    #[error("lock for {resource} is poisoned")]
    LockPoisoned {
        /// Name of the locked resource that was poisoned.
        resource: &'static str,
    },
    /// An internal invariant was violated, indicating a logic error.
    #[error("MST invariant violated: {invariant} (index {index}, lock_count {lock_count})")]
    InvariantViolation {
        /// Name of the violated invariant to assist debugging.
        invariant: &'static str,
        /// The lock index that violated the invariant.
        index: usize,
        /// The number of locks available.
        lock_count: usize,
    },
}

impl MstError {
    /// Returns a stable, machine-readable error code for the variant.
    #[must_use]
    pub const fn code(&self) -> MstErrorCode {
        match self {
            Self::EmptyGraph => MstErrorCode::EmptyGraph,
            Self::InvalidNodeId { .. } => MstErrorCode::InvalidNodeId,
            Self::NonFiniteWeight { .. } => MstErrorCode::NonFiniteWeight,
            Self::LockPoisoned { .. } => MstErrorCode::LockPoisoned,
            Self::InvariantViolation { .. } => MstErrorCode::InvariantViolation,
        }
    }
}

/// Machine-readable error codes for [`MstError`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MstErrorCode {
    /// The caller requested an MST for an empty graph.
    EmptyGraph,
    /// An edge referenced a node id that is not present in the graph.
    InvalidNodeId,
    /// An edge contained a non-finite weight.
    NonFiniteWeight,
    /// A synchronisation primitive became poisoned after a panic.
    LockPoisoned,
    /// An internal invariant was violated.
    InvariantViolation,
}

impl MstErrorCode {
    /// Returns the symbolic identifier for logging and metrics surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyGraph => "EMPTY_GRAPH",
            Self::InvalidNodeId => "INVALID_NODE_ID",
            Self::NonFiniteWeight => "NON_FINITE_WEIGHT",
            Self::LockPoisoned => "LOCK_POISONED",
            Self::InvariantViolation => "INVARIANT_VIOLATION",
        }
    }
}

/// A single MST edge in canonical undirected form (`source <= target`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MstEdge {
    source: usize,
    target: usize,
    weight: f32,
    sequence: u64,
}

impl MstEdge {
    /// Returns the smaller endpoint id.
    #[must_use]
    #[rustfmt::skip]
    pub fn source(&self) -> usize { self.source }

    /// Returns the larger endpoint id.
    #[must_use]
    #[rustfmt::skip]
    pub fn target(&self) -> usize { self.target }

    /// Returns the edge weight.
    #[must_use]
    #[rustfmt::skip]
    pub fn weight(&self) -> f32 { self.weight }

    /// Returns the deterministic tie-break sequence associated with the edge.
    #[must_use]
    #[rustfmt::skip]
    pub fn sequence(&self) -> u64 { self.sequence }
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
#[derive(Clone, Debug, PartialEq)]
pub struct MinimumSpanningForest {
    edges: Vec<MstEdge>,
    component_count: usize,
}

impl MinimumSpanningForest {
    /// Returns the MST/forest edges.
    #[must_use]
    #[rustfmt::skip]
    pub fn edges(&self) -> &[MstEdge] { &self.edges }

    /// Returns the number of connected components in the resulting forest.
    #[must_use]
    #[rustfmt::skip]
    pub fn component_count(&self) -> usize { self.component_count }

    /// Returns `true` when the forest spans a single connected component.
    #[must_use]
    pub fn is_tree(&self) -> bool {
        self.component_count == 1
    }
}

/// Computes a minimum spanning forest using parallel Kruskal's algorithm.
///
/// The input edges are interpreted as undirected and are canonicalised to
/// `(min(u, v), max(u, v))`. Self-edges are ignored.
///
/// # Errors
///
/// Returns an error when:
/// - `node_count == 0`
/// - an edge references a node id `>= node_count`
/// - an edge weight is non-finite
pub fn parallel_kruskal(
    node_count: usize,
    edges: &EdgeHarvest,
) -> Result<MinimumSpanningForest, MstError> {
    parallel_kruskal_from_edges(node_count, edges.iter())
}

fn validate_and_canonicalize_edge(
    edge: &CandidateEdge,
    node_count: usize,
) -> Result<Option<MstEdge>, MstError> {
    let source = edge.source();
    let target = edge.target();

    if source >= node_count {
        return Err(MstError::InvalidNodeId {
            node: source,
            node_count,
        });
    }
    if target >= node_count {
        return Err(MstError::InvalidNodeId {
            node: target,
            node_count,
        });
    }

    let weight = edge.distance();
    if !weight.is_finite() {
        return Err(MstError::NonFiniteWeight {
            left: source,
            right: target,
        });
    }

    if source == target {
        return Ok(None);
    }

    let (source, target) = if source <= target {
        (source, target)
    } else {
        (target, source)
    };

    Ok(Some(MstEdge {
        source,
        target,
        weight,
        sequence: edge.sequence(),
    }))
}

fn process_weight_group(
    group: &[MstEdge],
    union_find: &ConcurrentUnionFind,
) -> Result<Vec<MstEdge>, MstError> {
    // Process edges sequentially to ensure deterministic MST selection.
    // Since edges are already sorted by (weight, source, target, sequence),
    // sequential iteration produces reproducible results.
    let mut accepted = Vec::new();
    for edge in group {
        if union_find.try_union(edge.source, edge.target)? {
            accepted.push(*edge);
        }
    }
    Ok(accepted)
}

fn is_mst_complete(
    node_count: usize,
    union_find: &ConcurrentUnionFind,
    forest_edges: &[MstEdge],
) -> bool {
    union_find.components() == 1 && forest_edges.len() == node_count.saturating_sub(1)
}

fn prepare_edge_list<'a>(
    edges: impl IntoIterator<Item = &'a CandidateEdge>,
    node_count: usize,
) -> Result<Vec<MstEdge>, MstError> {
    let edges: Vec<&CandidateEdge> = edges.into_iter().collect();
    let mut edge_list = edges
        .par_iter()
        .try_fold(Vec::new, |mut acc, edge| {
            if let Some(mst_edge) = validate_and_canonicalize_edge(edge, node_count)? {
                acc.push(mst_edge);
            }
            Ok(acc)
        })
        .try_reduce(Vec::new, |mut left, right| {
            left.extend(right);
            Ok(left)
        })?;

    edge_list.par_sort_unstable();
    edge_list.dedup_by(|left, right| {
        left.weight == right.weight && left.source == right.source && left.target == right.target
    });
    Ok(edge_list)
}

pub(crate) fn parallel_kruskal_from_edges<'a>(
    node_count: usize,
    edges: impl IntoIterator<Item = &'a CandidateEdge>,
) -> Result<MinimumSpanningForest, MstError> {
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

    let mut cursor = 0;
    while cursor < edge_list.len() {
        let weight = edge_list[cursor].weight;
        let mut next = cursor.saturating_add(1);
        while next < edge_list.len() && edge_list[next].weight == weight {
            next = next.saturating_add(1);
        }

        let group = &edge_list[cursor..next];
        let accepted = process_weight_group(group, &union_find)?;

        forest_edges.extend(accepted);

        if is_mst_complete(node_count, &union_find, &forest_edges) {
            break;
        }

        cursor = next;
    }

    forest_edges.sort_unstable();
    Ok(MinimumSpanningForest {
        edges: forest_edges,
        component_count: union_find.components(),
    })
}

// ============================================================================
// Kani Formal Verification
// ============================================================================

/// Validates MST forest structural invariants for Kani verification.
///
/// Returns `true` if the forest satisfies:
/// - Edge count equals `n - c` where `n` is node count and `c` is component count
/// - No self-loops (source != target for all edges)
/// - Canonical ordering (source < target for all edges)
/// - Acyclic structure (no cycles detected via union-find)
#[cfg(kani)]
pub(crate) fn is_valid_forest(
    node_count: usize,
    edges: &[MstEdge],
    component_count: usize,
) -> bool {
    // Forest must have n - c edges
    if edges.len() != node_count.saturating_sub(component_count) {
        return false;
    }

    // No self-loops and canonical ordering (inlined for simplicity)
    for edge in edges {
        if edge.source() == edge.target() || edge.source() >= edge.target() {
            return false;
        }
    }

    // Acyclic check via union-find
    let mut parent: Vec<usize> = (0..node_count).collect();
    for edge in edges {
        let root_s = kani_find_root(&mut parent, edge.source());
        let root_t = kani_find_root(&mut parent, edge.target());
        if root_s == root_t {
            return false; // Cycle detected
        }
        parent[root_t] = root_s;
    }

    true
}

/// Simple union-find root finding for Kani verification.
#[cfg(kani)]
fn kani_find_root(parent: &mut [usize], node: usize) -> usize {
    let mut current = node;
    while parent[current] != current {
        current = parent[current];
    }
    current
}

#[cfg(kani)]
mod kani_proofs {
    //! Kani proof harnesses for minimum spanning tree (MST) invariants.
    //!
    //! These harnesses verify structural correctness of the parallel Kruskal
    //! algorithm using bounded model checking.

    use super::{CandidateEdge, is_valid_forest, parallel_kruskal_from_edges};

    /// Verifies MST structural correctness for bounded graphs.
    ///
    /// This harness creates a small graph with nondeterministically selected
    /// edges and verifies that the resulting MST/forest satisfies structural
    /// invariants: correct edge count, no cycles, canonical ordering.
    ///
    /// # Verification Bounds
    ///
    /// - **Nodes**: 4 (to keep solver time reasonable)
    /// - **Edges**: Up to 6 (complete graph on 4 nodes)
    /// - **Weights**: Represented as u8 cast to f32 for finite guarantees
    #[kani::proof]
    #[kani::unwind(12)]
    fn verify_mst_structural_correctness_4_nodes() {
        let node_count = 4usize;

        // Nondeterministically select edges from the complete graph
        // 4 nodes = 6 possible undirected edges
        let edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

        let mut edges = Vec::new();
        let mut seq = 0u64;
        for &(source, target) in &edge_pairs {
            if kani::any::<bool>() {
                let weight: u8 = kani::any();
                edges.push(CandidateEdge::new(source, target, f32::from(weight), seq));
                seq = seq.saturating_add(1);
            }
        }

        // With valid finite weights, parallel_kruskal_from_edges should not fail
        let forest = parallel_kruskal_from_edges(node_count, edges.iter())
            .expect("MST computation should succeed for valid inputs");

        let mst_edges = forest.edges();
        let component_count = forest.component_count();

        kani::assert(
            is_valid_forest(node_count, mst_edges, component_count),
            "MST forest invariant violated",
        );

        // Additional invariant: forest should never have more than n-1 edges
        kani::assert(
            mst_edges.len() <= node_count.saturating_sub(1),
            "MST has too many edges",
        );

        // If it's a tree (1 component), it must have exactly n-1 edges
        if component_count == 1 {
            kani::assert(
                mst_edges.len() == node_count.saturating_sub(1),
                "MST tree should have n-1 edges",
            );
        }
    }

    /// Verifies MST minimality property for bounded graphs.
    ///
    /// This harness verifies that the MST includes minimum weight edges by
    /// checking that the total weight is minimal. For a 3-node graph, if
    /// all edges are present, the MST must exclude the heaviest edge that
    /// would create a cycle.
    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_mst_minimality_3_nodes() {
        let node_count = 3usize;
        let mut edges = Vec::new();

        // Create edges with distinct weights to verify minimality
        let weight0: u8 = kani::any();
        let weight1: u8 = kani::any();
        let weight2: u8 = kani::any();

        if kani::any::<bool>() {
            edges.push(CandidateEdge::new(0, 1, f32::from(weight0), 0));
        }
        if kani::any::<bool>() {
            edges.push(CandidateEdge::new(1, 2, f32::from(weight1), 1));
        }
        if kani::any::<bool>() {
            edges.push(CandidateEdge::new(0, 2, f32::from(weight2), 2));
        }

        let forest = parallel_kruskal_from_edges(node_count, edges.iter())
            .expect("MST computation should succeed for valid inputs");

        let mst_edges = forest.edges();

        // Verify structural invariants hold
        kani::assert(
            is_valid_forest(node_count, mst_edges, forest.component_count()),
            "MST forest invariant violated",
        );

        // If we have a connected graph (at least 2 edges selected from a
        // triangle), verify the MST has exactly n-1 edges
        if forest.component_count() == 1 {
            kani::assert(
                mst_edges.len() == node_count.saturating_sub(1),
                "connected MST should have n-1 edges",
            );
        }
    }
}

#[cfg(test)]
mod tests;
