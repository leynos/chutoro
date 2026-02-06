//! Property 2: Structural invariant verification.
//!
//! For any MST/forest produced by parallel Kruskal, verifies:
//!
//! - **Acyclicity** — no cycles (union-find based detection).
//! - **Connectivity** — connected input produces connected output.
//! - **Edge count** — `V - C` edges for `C` connected components.
//! - **No self-loops** — `source != target` for all edges.
//! - **Canonical form** — `source < target` for all edges.
//! - **Finite weights** — all edge weights are finite.
//!
//! See `docs/property-testing-design.md` §4.3.2.

use proptest::test_runner::{TestCaseError, TestCaseResult};

use crate::{EdgeHarvest, parallel_kruskal};

use super::types::MstFixture;

/// Runs the structural invariant property for the given fixture.
pub(super) fn run_structural_invariants_property(fixture: &MstFixture) -> TestCaseResult {
    let harvest = EdgeHarvest::new(fixture.edges.clone());

    let result = parallel_kruskal(fixture.node_count, &harvest).map_err(|e| {
        TestCaseError::fail(format!(
            "parallel_kruskal failed: {e} (distribution={:?}, nodes={}, edges={})",
            fixture.distribution,
            fixture.node_count,
            fixture.edges.len(),
        ))
    })?;

    let mst_edges = result.edges();
    let node_count = fixture.node_count;

    // Canonical form: source < target (strict, since self-loops are excluded).
    for (i, edge) in mst_edges.iter().enumerate() {
        if edge.source() >= edge.target() {
            return Err(TestCaseError::fail(format!(
                "edge {i}: not canonical ({} >= {})",
                edge.source(),
                edge.target(),
            )));
        }
    }

    // No self-loops.
    for (i, edge) in mst_edges.iter().enumerate() {
        if edge.source() == edge.target() {
            return Err(TestCaseError::fail(format!(
                "edge {i}: self-loop on node {}",
                edge.source(),
            )));
        }
    }

    // Finite weights.
    for (i, edge) in mst_edges.iter().enumerate() {
        if !edge.weight().is_finite() {
            return Err(TestCaseError::fail(format!(
                "edge {i}: non-finite weight {}",
                edge.weight(),
            )));
        }
    }

    // Acyclicity check via union-find.
    let mut parent: Vec<usize> = (0..node_count).collect();
    for (i, edge) in mst_edges.iter().enumerate() {
        let ra = find_root(&mut parent, edge.source());
        let rb = find_root(&mut parent, edge.target());
        if ra == rb {
            return Err(TestCaseError::fail(format!(
                "edge {i}: ({}, {}) creates a cycle",
                edge.source(),
                edge.target(),
            )));
        }
        parent[rb] = ra;
    }

    // Edge count: forest must have n - c edges.
    let expected_edges = node_count.saturating_sub(result.component_count());
    if mst_edges.len() != expected_edges {
        return Err(TestCaseError::fail(format!(
            "edge count {}, expected n - c = {} (n={node_count}, c={})",
            mst_edges.len(),
            expected_edges,
            result.component_count(),
        )));
    }

    // Connectivity: if the input is connected, the output must be a tree.
    let input_components = count_input_components(fixture);
    if input_components == 1 && !result.is_tree() {
        return Err(TestCaseError::fail(format!(
            "input is connected but output has {} components",
            result.component_count(),
        )));
    }

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Path-compressing find for the verification union-find.
fn find_root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

/// Counts connected components in the input graph by applying union-find
/// over the fixture's raw edges (ignoring self-loops, out-of-bounds, and
/// non-finite weights).
fn count_input_components(fixture: &MstFixture) -> usize {
    let n = fixture.node_count;
    if n == 0 {
        return 0;
    }

    let mut parent: Vec<usize> = (0..n).collect();
    let mut components = n;

    for edge in &fixture.edges {
        let s = edge.source();
        let t = edge.target();
        if s == t || s >= n || t >= n || !edge.distance().is_finite() {
            continue;
        }
        let ra = find_root(&mut parent, s);
        let rb = find_root(&mut parent, t);
        if ra != rb {
            parent[rb] = ra;
            components -= 1;
        }
    }

    components
}
