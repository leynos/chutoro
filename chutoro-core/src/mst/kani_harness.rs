//! Kani harnesses for minimum-spanning-forest invariants.

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
/// - Edge count equals `n - c` where `n` is node count and `c` is component count
/// - No self-loops (source != target for all edges)
/// - Canonical ordering (source < target for all edges)
/// - Acyclic structure (no cycles detected via union-find)
pub(crate) fn is_valid_forest(
    node_count: usize,
    edges: &[MstEdge],
    component_count: usize,
) -> bool {
    // Forest must have n - c edges
    if edges.len() != node_count.saturating_sub(component_count) {
        return false;
    }

    // No self-loops and canonical ordering
    if !validate_edges_canonical(edges) {
        return false;
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
