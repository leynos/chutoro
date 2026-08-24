//! CPU minimum spanning tree (MST) construction.
//!
//! This module provides a parallel Kruskal implementation intended for CPU
//! backends. The algorithm parallelizes the global edge sort via Rayon and
//! performs concurrent cycle checks using a striped-lock union-find.

mod union_find;

#[cfg(any(kani, test))]
mod kani_model;
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
    /// A synchronization primitive became poisoned after a panic.
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
    /// A synchronization primitive became poisoned after a panic.
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
/// - `node_count == 0`
/// - an edge references a node id `>= node_count`
/// - an edge weight is non-finite
pub fn parallel_kruskal(
    node_count: usize,
    edges: &EdgeHarvest,
) -> Result<MinimumSpanningForest, MstError> {
    parallel_kruskal_from_edges(node_count, edges.iter())
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

/// Accept non-cycling edges from one equal-weight group deterministically.
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

/// Report whether a forest contains the edge count of a spanning tree.
fn is_mst_complete(
    node_count: usize,
    union_find: &ConcurrentUnionFind,
    forest_edges: &[MstEdge],
) -> bool {
    union_find.components() == 1 && forest_edges.len() == node_count.saturating_sub(1)
}

/// Validate, canonicalise, sort, and deduplicate candidate edges.
fn prepare_edge_list<'a>(
    edges: impl IntoIterator<Item = &'a CandidateEdge>,
    node_count: usize,
) -> Result<Vec<MstEdge>, MstError> {
    let candidate_edges: Vec<&CandidateEdge> = edges.into_iter().collect();
    let mut edge_list = candidate_edges
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
        matches!(
            left.weight.partial_cmp(&right.weight),
            Some(Ordering::Equal)
        ) && left.source == right.source
            && left.target == right.target
    });
    Ok(edge_list)
}

/// Run parallel Kruskal directly over an iterator of candidate edges.
pub(crate) fn parallel_kruskal_from_edges<'a>(
    node_count: usize,
    edges: impl IntoIterator<Item = &'a CandidateEdge>,
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
            let accepted = process_weight_group(group, &union_find)?;

            forest_edges.extend(accepted);

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
mod property;

#[cfg(test)]
mod tests;
