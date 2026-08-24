//! Fast-tier Kani harnesses for minimum-spanning-forest invariants.
//!
//! Both harnesses run under `make kani` and the nightly `make kani-full`
//! package sweep.

use super::*;

// ============================================================================
// Kani Formal Verification
// ============================================================================

/// Returns `true` if all edges have canonical ordering and no self-loops.
///
/// Canonical ordering requires `source < target` for all edges.
fn validate_edges_canonical(edges: &[MstEdge]) -> bool {
    edges
        .iter()
        .all(|edge| edge.source() != edge.target() && edge.source() < edge.target())
}

/// Validates MST forest structural invariants for Kani verification.
///
/// Returns `true` if the forest satisfies:
/// - At most four nodes (the harness bound)
/// - Edge count equals `n - c` where `n` is node count and `c` is component count
/// - No self-loops and canonical ordering (`source < target` for all edges)
/// - Acyclic structure (no cycles detected via union-find)
///
/// The enclosing module is compiled only under `cfg(kani)`, so no per-item
/// cfg gate is needed.
pub(crate) fn is_valid_forest(
    node_count: usize,
    edges: &[MstEdge],
    component_count: usize,
) -> bool {
    if node_count > 4 || edges.len() != node_count.saturating_sub(component_count) {
        return false;
    }
    if !validate_edges_canonical(edges) {
        return false;
    }

    let mut parent = [0, 1, 2, 3];
    for edge in edges {
        let root_s = kani_find_root(&mut parent, edge.source());
        let root_t = kani_find_root(&mut parent, edge.target());
        if root_s == root_t {
            return false;
        }
        parent[root_t] = root_s;
    }

    true
}

/// Simple union-find root finding for Kani verification.
fn kani_find_root(parent: &mut [usize], node: usize) -> usize {
    let mut current = node;
    while parent[current] != current {
        current = parent[current];
    }
    current
}

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
    /// - **Weights**: A fixed finite representative, as weights do not affect
    ///   the structural invariants
    #[kani::proof]
    #[kani::solver(kissat)]
    #[kani::unwind(12)]
    fn verify_mst_structural_correctness_4_nodes() {
        let node_count = 4usize;

        // Nondeterministically select edges from the complete graph
        // 4 nodes = 6 possible undirected edges
        let edges = [
            selected_candidate(0, 1, 0, 0),
            selected_candidate(0, 2, 0, 1),
            selected_candidate(0, 3, 0, 2),
            selected_candidate(1, 2, 0, 3),
            selected_candidate(1, 3, 0, 4),
            selected_candidate(2, 3, 0, 5),
        ];

        // With valid finite weights, parallel_kruskal_from_edges should not fail
        let Ok(forest) = parallel_kruskal_from_edges(node_count, edges.iter()) else {
            kani::assert(false, "MST computation should succeed for valid inputs");
            return;
        };

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
    /// This harness verifies that the forest is weight-minimal, not merely
    /// structurally valid. It draws the edge selection explicitly, computes
    /// the expected minimal spanning weight for that selection, and asserts
    /// that the returned forest matches it. When the full triangle is
    /// selected, this forces the heaviest edge (weight 2) to be excluded.
    #[kani::proof]
    #[kani::solver(kissat)]
    #[kani::unwind(10)]
    fn verify_mst_minimality_3_nodes() {
        let node_count = 3usize;
        let select_01 = kani::any::<bool>();
        let select_12 = kani::any::<bool>();
        let select_02 = kani::any::<bool>();
        let edges = [
            candidate_or_self_loop(0, 1, 0, 0, select_01),
            candidate_or_self_loop(1, 2, 1, 1, select_12),
            candidate_or_self_loop(0, 2, 2, 2, select_02),
        ];

        let Ok(forest) = parallel_kruskal_from_edges(node_count, edges.iter()) else {
            kani::assert(false, "MST computation should succeed for valid inputs");
            return;
        };

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

        // Verify minimality: the forest's total weight must equal the
        // minimal spanning weight for the selected edge subset. Weights are
        // small integers, so f32 summation is exact.
        let total_weight: f32 = mst_edges.iter().map(|edge| edge.weight()).sum();
        let expected = expected_minimal_weight(select_01, select_12, select_02);
        kani::assert(
            total_weight == expected,
            "MST total weight must be minimal for the selected edges",
        );
    }

    /// Returns `true` when the weight-2 edge closes the triangle and must
    /// therefore be excluded from the minimal spanning tree.
    fn heavy_edge_is_redundant(select_01: bool, select_12: bool, select_02: bool) -> bool {
        select_01 && select_12 && select_02
    }

    /// Returns the minimal spanning weight of the triangle subset where edge
    /// (0,1) weighs 0, edge (1,2) weighs 1, and edge (0,2) weighs 2.
    ///
    /// Edge (0,1) contributes nothing to the total. Selected edges cannot
    /// otherwise form a cycle, so the forest is exactly the selected edges
    /// unless the heavy edge closes the triangle and is excluded.
    fn expected_minimal_weight(select_01: bool, select_12: bool, select_02: bool) -> f32 {
        let mut weight = 0.0;
        if select_12 {
            weight += 1.0;
        }
        if select_02 && !heavy_edge_is_redundant(select_01, select_12, select_02) {
            weight += 2.0;
        }
        weight
    }

    fn selected_candidate(
        source: usize,
        target: usize,
        weight: u8,
        sequence: u64,
    ) -> CandidateEdge {
        candidate_or_self_loop(source, target, weight, sequence, kani::any::<bool>())
    }

    /// Builds the edge when `selected`, or a same-weight self-loop (which
    /// validation discards) when not.
    fn candidate_or_self_loop(
        source: usize,
        target: usize,
        weight: u8,
        sequence: u64,
        selected: bool,
    ) -> CandidateEdge {
        if selected {
            CandidateEdge::new(source, target, f32::from(weight), sequence)
        } else {
            CandidateEdge::new(source, source, f32::from(weight), sequence)
        }
    }
}
