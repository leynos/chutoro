//! Sequential Kruskal oracle for MST property verification.
//!
//! Provides a simple, trusted, sequential implementation of Kruskal's
//! algorithm for use as a reference oracle in property tests.  The sort
//! order intentionally mirrors `MstEdge::Ord` so that total-weight
//! comparisons are valid.

use std::cmp::Ordering;

use crate::CandidateEdge;

use super::helpers::{find_root, is_invalid_edge};

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
    let accepted = sequential_kruskal_edges(node_count, edges);
    // `Iterator::sum` starts from `-0.0`, the additive identity for floats.
    // Folding from positive zero keeps an empty forest's total at `0.0`, which
    // is the value the oracle has always reported and `total_cmp` distinguishes.
    let total_weight = accepted.iter().fold(0.0_f64, |total, (_, _, weight, _)| {
        std::ops::Add::add(total, f64::from(*weight))
    });
    let edge_count = accepted.len();

    SequentialMstResult {
        total_weight,
        edge_count,
        component_count: node_count.saturating_sub(edge_count),
    }
}

/// Computes a minimum spanning forest and returns the accepted edges.
///
/// Each entry is `(source, target, weight, sequence)` in canonical undirected
/// form, ordered exactly as the oracle accepts it. Callers outside this module
/// use the list to compare a production forest with the reference without
/// needing to construct [`crate::MstEdge`] values.
pub(crate) fn sequential_kruskal_edges(
    node_count: usize,
    edges: &[CandidateEdge],
) -> Vec<(usize, usize, f32, u64)> {
    if node_count == 0 || edges.is_empty() {
        return Vec::new();
    }

    let mut canon = canonicalize_and_filter(edges, node_count);
    canon.sort_unstable_by(cmp_canon_edge);
    dedup_canon_edges(&mut canon);

    let mut parent: Vec<usize> = (0..node_count).collect();
    let mut rank: Vec<usize> = vec![0; node_count];
    let mut accepted = Vec::new();

    for edge in &canon {
        let ra = find_root(&mut parent, edge.source);
        let rb = find_root(&mut parent, edge.target);
        if ra != rb {
            union_by_rank(&mut parent, &mut rank, ra, rb);
            accepted.push((edge.source, edge.target, edge.weight, edge.sequence));
        }
    }

    accepted
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

/// Canonicalizes edges to `(min, max)`, filtering out self-loops and
/// out-of-bounds references.
fn canonicalize_and_filter(edges: &[CandidateEdge], node_count: usize) -> Vec<CanonEdge> {
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
        left.weight.total_cmp(&right.weight).is_eq()
            && left.source == right.source
            && left.target == right.target
    });
}

/// Selects the root and child for a union operation.
///
/// Prefers the node with the higher rank; when ranks are equal, the
/// smaller index becomes root to ensure deterministic tie-breaking.
fn choose_root(rank: &[usize], a: usize, b: usize) -> (usize, usize) {
    let Some(&a_rank) = rank.get(a) else {
        return (b, a);
    };
    let Some(&b_rank) = rank.get(b) else {
        return (a, b);
    };
    match a_rank.cmp(&b_rank) {
        Ordering::Greater => (a, b),
        Ordering::Less => (b, a),
        Ordering::Equal => {
            if a <= b {
                (a, b)
            } else {
                (b, a)
            }
        }
    }
}

/// Union by rank, breaking ties by smaller index.
fn union_by_rank(parent: &mut [usize], rank: &mut [usize], a: usize, b: usize) {
    let (root, child) = choose_root(rank, a, b);
    let Some(root_rank) = rank.get(root).copied() else {
        return;
    };
    let Some(child_rank) = rank.get(child).copied() else {
        return;
    };
    let Some(child_parent) = parent.get_mut(child) else {
        return;
    };
    *child_parent = root;
    if root_rank == child_rank
        && let Some(root_rank_slot) = rank.get_mut(root)
    {
        *root_rank_slot = root_rank.saturating_add(1);
    }
}
