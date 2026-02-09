//! Sequential Kruskal oracle for MST property verification.
//!
//! Provides a simple, trusted, sequential implementation of Kruskal's
//! algorithm for use as a reference oracle in property tests.  The sort
//! order intentionally mirrors `MstEdge::Ord` so that total-weight
//! comparisons are valid.

use std::cmp::Ordering;

use crate::CandidateEdge;

use super::helpers::find_root;

/// Result of the sequential Kruskal oracle.
#[derive(Clone, Debug)]
pub(super) struct SequentialMstResult {
    /// Total weight of the MST/forest, accumulated as `f64` for precision.
    pub total_weight: f64,
    /// Number of edges in the MST/forest.
    pub edge_count: usize,
    /// Number of connected components after MST construction.
    pub component_count: usize,
}

/// Computes a minimum spanning forest using sequential Kruskal's algorithm.
///
/// The sort order matches `MstEdge::Ord`:
/// `(weight.total_cmp, source, target, sequence)` after canonicalization
/// and deduplication.  This ensures that the sequential oracle accepts the
/// same edges as the parallel implementation, yielding identical total
/// weights.
pub(super) fn sequential_kruskal(
    node_count: usize,
    edges: &[CandidateEdge],
) -> SequentialMstResult {
    if node_count == 0 || edges.is_empty() {
        return SequentialMstResult {
            total_weight: 0.0,
            edge_count: 0,
            component_count: node_count,
        };
    }

    let mut canon = canonicalise_and_filter(edges, node_count);
    canon.sort_unstable_by(cmp_canon_edge);
    dedup_canon_edges(&mut canon);

    let mut parent: Vec<usize> = (0..node_count).collect();
    let mut rank: Vec<usize> = vec![0; node_count];
    let mut components = node_count;
    let mut total_weight: f64 = 0.0;
    let mut edge_count: usize = 0;

    for edge in &canon {
        let ra = find_root(&mut parent, edge.source);
        let rb = find_root(&mut parent, edge.target);
        if ra != rb {
            union_by_rank(&mut parent, &mut rank, ra, rb);
            total_weight += f64::from(edge.weight);
            edge_count += 1;
            components -= 1;
        }
    }

    SequentialMstResult {
        total_weight,
        edge_count,
        component_count: components,
    }
}

// ── Internal types ──────────────────────────────────────────────────────

/// Canonicalized edge for oracle processing, mirroring `MstEdge` fields.
struct CanonEdge {
    source: usize,
    target: usize,
    weight: f32,
    sequence: u64,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Returns `true` when an edge should be excluded from MST consideration.
///
/// An edge is invalid when any of the following hold:
/// - it is a self-loop,
/// - either endpoint falls outside the node range, or
/// - its weight is non-finite (NaN / infinity).
fn is_invalid_edge(source: usize, target: usize, node_count: usize, weight: f32) -> bool {
    let is_self_loop = source == target;
    let is_out_of_bounds = source >= node_count || target >= node_count;
    let is_non_finite = !weight.is_finite();
    is_self_loop || is_out_of_bounds || is_non_finite
}

/// Canonicalizes edges to `(min, max)`, filtering out self-loops and
/// out-of-bounds references.
fn canonicalise_and_filter(edges: &[CandidateEdge], node_count: usize) -> Vec<CanonEdge> {
    edges
        .iter()
        .filter_map(|e| {
            let s = e.source();
            let t = e.target();
            if is_invalid_edge(s, t, node_count, e.distance()) {
                return None;
            }
            let (lo, hi) = if s <= t { (s, t) } else { (t, s) };
            Some(CanonEdge {
                source: lo,
                target: hi,
                weight: e.distance(),
                sequence: e.sequence(),
            })
        })
        .collect()
}

/// Sort comparator matching `MstEdge::Ord` exactly.
fn cmp_canon_edge(a: &CanonEdge, b: &CanonEdge) -> Ordering {
    a.weight
        .total_cmp(&b.weight)
        .then_with(|| a.source.cmp(&b.source))
        .then_with(|| a.target.cmp(&b.target))
        .then_with(|| a.sequence.cmp(&b.sequence))
}

/// Removes consecutive duplicate edges sharing the same
/// `(weight, source, target)`, keeping the one with the lowest sequence.
///
/// Mirrors the dedup logic in `prepare_edge_list`.
fn dedup_canon_edges(edges: &mut Vec<CanonEdge>) {
    edges.dedup_by(|right, left| {
        left.weight == right.weight && left.source == right.source && left.target == right.target
    });
}

/// Selects the root and child for a union operation.
///
/// Prefers the node with the higher rank; when ranks are equal, the
/// smaller index becomes root to ensure deterministic tie-breaking.
fn choose_root(rank: &[usize], a: usize, b: usize) -> (usize, usize) {
    match rank[a].cmp(&rank[b]) {
        std::cmp::Ordering::Greater => (a, b),
        std::cmp::Ordering::Less => (b, a),
        std::cmp::Ordering::Equal if a <= b => (a, b),
        std::cmp::Ordering::Equal => (b, a),
    }
}

/// Union by rank, breaking ties by smaller index.
fn union_by_rank(parent: &mut [usize], rank: &mut [usize], a: usize, b: usize) {
    let (root, child) = choose_root(rank, a, b);
    parent[child] = root;
    if rank[root] == rank[child] {
        rank[root] += 1;
    }
}
