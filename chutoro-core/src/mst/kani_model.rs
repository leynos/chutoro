//! Sequential Kani model for parallel Kruskal execution.
//!
//! Kani models concurrency as sequential execution. This model preserves the
//! production algorithm's validated edge ordering and deterministic union
//! selection while omitting Rayon and synchronisation internals that do not
//! contribute to the forest invariants proved by the bounded harnesses.
//! It is compiled only for Kani harnesses and for the exhaustive
//! model-equivalence tests; production callers must not use it.

#![expect(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::missing_const_for_fn,
    clippy::shadow_reuse,
    reason = "the proof model uses fixed arrays and direct indexing to keep CBMC's symbolic formula tractable"
)]

use crate::CandidateEdge;

#[cfg(kani)]
use super::MinimumSpanningForest;
use super::{MstEdge, MstError, validate_and_canonicalize_edge};

/// Placeholder edge used to initialise the bounded forest buffer.
const EMPTY_MST_EDGE: MstEdge = MstEdge {
    source: 0,
    target: 0,
    weight: 0.0,
    sequence: 0,
};

/// Bounded forest produced by the sequential Kani model.
///
/// The representation is cfg-independent so the equivalence tests can compare
/// it directly against the production forest.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ModelForest {
    edges: [MstEdge; 3],
    edge_count: usize,
    component_count: usize,
}

impl ModelForest {
    /// Returns the accepted forest edges in sorted order.
    #[cfg(test)]
    #[rustfmt::skip]
    pub(super) fn edges(&self) -> &[MstEdge] { &self.edges[..self.edge_count] }

    /// Returns the number of connected components in the resulting forest.
    #[cfg(test)]
    #[rustfmt::skip]
    pub(super) fn component_count(&self) -> usize { self.component_count }
}

/// Computes the Kani-only sequential model of parallel Kruskal.
#[cfg(kani)]
pub(super) fn parallel_kruskal_from_edges_for_kani<'a>(
    node_count: usize,
    edges: impl IntoIterator<Item = &'a CandidateEdge>,
) -> Result<MinimumSpanningForest, MstError> {
    let forest = kruskal_model(node_count, edges)?;
    Ok(MinimumSpanningForest {
        edges: forest.edges,
        edge_count: forest.edge_count,
        component_count: forest.component_count,
    })
}

/// Runs the bounded sequential Kruskal model shared by Kani and the
/// equivalence tests.
pub(super) fn kruskal_model<'a>(
    node_count: usize,
    edges: impl IntoIterator<Item = &'a CandidateEdge>,
) -> Result<ModelForest, MstError> {
    if node_count == 0 {
        return Err(MstError::EmptyGraph);
    }

    if node_count > 4 {
        return Err(MstError::InvariantViolation {
            invariant: "Kani MST model supports at most four nodes",
            index: node_count,
            lock_count: 4,
        });
    }

    let mut edge_list = [None; 6];
    let mut edge_count = 0;
    for edge in edges {
        if let Some(edge) = validate_and_canonicalize_edge(edge, node_count)? {
            let Some(slot) = edge_list.get_mut(edge_count) else {
                return Err(MstError::InvariantViolation {
                    invariant: "Kani MST model supports at most six edges",
                    index: edge_count,
                    lock_count: edge_list.len(),
                });
            };
            *slot = Some(edge);
            edge_count += 1;
        }
    }
    sort_edges_for_kani(&mut edge_list, edge_count);
    edge_count = deduplicate_edges_for_kani(&mut edge_list, edge_count);

    let mut parents = [0, 1, 2, 3];
    let mut ranks = [0; 4];
    let mut node = 0;
    while node < node_count {
        parents[node] = node;
        node += 1;
    }
    let mut component_count = node_count;
    let mut forest_edges = [EMPTY_MST_EDGE; 3];
    let mut forest_edge_count = 0;

    let mut index = 0;
    while index < edge_count {
        let Some(edge) = edge_list[index] else {
            return Err(MstError::InvariantViolation {
                invariant: "Kani MST edge buffer must be populated",
                index,
                lock_count: edge_count,
            });
        };
        let source_root = find_root(&mut parents, edge.source);
        let target_root = find_root(&mut parents, edge.target);
        if source_root != target_root {
            union_roots(&mut parents, &mut ranks, source_root, target_root);
            component_count = component_count.saturating_sub(1);
            forest_edges[forest_edge_count] = edge;
            forest_edge_count += 1;
        }

        if component_count == 1 && forest_edge_count == node_count.saturating_sub(1) {
            break;
        }
        index += 1;
    }

    sort_forest_edges_for_kani(&mut forest_edges, forest_edge_count);
    Ok(ModelForest {
        edges: forest_edges,
        edge_count: forest_edge_count,
        component_count,
    })
}

/// Insertion-sorts the populated prefix of the bounded edge buffer.
fn sort_edges_for_kani(edges: &mut [Option<MstEdge>], edge_count: usize) {
    let mut index = 1;
    while index < edge_count {
        let mut current = index;
        while current > 0 && should_swap_edges(edges, current) {
            edges.swap(current, current - 1);
            current -= 1;
        }
        index += 1;
    }
}

/// Compacts adjacent duplicates out of the sorted prefix.
///
/// Returns the number of unique edges retained.
fn deduplicate_edges_for_kani(edges: &mut [Option<MstEdge>], edge_count: usize) -> usize {
    let mut unique_count = 0;
    let mut index = 0;
    while index < edge_count {
        if let Some(edge) = edges[index] {
            let is_duplicate = unique_count > 0 && duplicate_edges(edges[unique_count - 1], edge);
            if !is_duplicate {
                edges[unique_count] = Some(edge);
                unique_count += 1;
            }
        }
        index += 1;
    }
    unique_count
}

/// Insertion-sorts the accepted forest edges into canonical order.
fn sort_forest_edges_for_kani(edges: &mut [MstEdge], edge_count: usize) {
    let mut index = 1;
    while index < edge_count {
        let mut current = index;
        while current > 0 && edges[current] < edges[current - 1] {
            edges.swap(current, current - 1);
            current -= 1;
        }
        index += 1;
    }
}

/// Reports whether adjacent populated slots are out of order.
///
/// Pairs involving an empty slot never swap; the sorted prefix is fully
/// populated at the only call site, so that arm is defensive.
fn should_swap_edges(edges: &[Option<MstEdge>], current: usize) -> bool {
    match (edges[current], edges[current - 1]) {
        (Some(current), Some(previous)) => current < previous,
        _ => false,
    }
}

/// Reports whether two edges share weight and canonical endpoints.
fn duplicate_edges(left: Option<MstEdge>, right: MstEdge) -> bool {
    left.is_some_and(|left| {
        left.weight == right.weight && left.source == right.source && left.target == right.target
    })
}

/// Finds the union-find root of `node`, halving paths as it walks.
fn find_root(parents: &mut [usize; 4], node: usize) -> usize {
    let mut current = node;
    while parents[current] != current {
        let parent = parents[current];
        let grandparent = parents[parent];
        if grandparent != parent {
            parents[current] = grandparent;
        }
        current = parent;
    }
    current
}

/// Unions two roots by rank, breaking ties towards the smaller id.
fn union_roots(parents: &mut [usize; 4], ranks: &mut [usize; 4], left: usize, right: usize) {
    let (parent, child) = if ranks[left] > ranks[right] {
        (left, right)
    } else if ranks[right] > ranks[left] {
        (right, left)
    } else if left <= right {
        (left, right)
    } else {
        (right, left)
    };

    parents[child] = parent;
    if ranks[left] == ranks[right] {
        ranks[parent] = ranks[parent].saturating_add(1);
    }
}
