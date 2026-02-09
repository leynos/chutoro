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

use crate::{EdgeHarvest, MstEdge, parallel_kruskal};

use super::helpers::{find_root, is_invalid_edge};
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

    validate_canonical_form(mst_edges)?;
    validate_no_self_loops(mst_edges)?;
    validate_finite_weights(mst_edges)?;
    validate_acyclicity(fixture.node_count, mst_edges)?;
    validate_edge_count(
        fixture.node_count,
        mst_edges.len(),
        result.component_count(),
    )?;
    validate_connectivity(fixture, &result)?;

    Ok(())
}

/// Generic edge validator that applies a predicate to each edge, returning
/// early with an error if the predicate produces a message.
fn validate_edges<F>(edges: &[MstEdge], mut predicate: F) -> TestCaseResult
where
    F: FnMut(usize, &MstEdge) -> Option<String>,
{
    for (i, edge) in edges.iter().enumerate() {
        if let Some(msg) = predicate(i, edge) {
            return Err(TestCaseError::fail(msg));
        }
    }
    Ok(())
}

// ── Validation helpers ──────────────────────────────────────────────────

/// Verifies that every MST edge is in canonical form (`source < target`).
fn validate_canonical_form(edges: &[MstEdge]) -> TestCaseResult {
    validate_edges(edges, |i, edge| {
        (edge.source() >= edge.target()).then(|| {
            format!(
                "edge {i}: not canonical ({} >= {})",
                edge.source(),
                edge.target(),
            )
        })
    })
}

/// Verifies that no MST edge is a self-loop.
fn validate_no_self_loops(edges: &[MstEdge]) -> TestCaseResult {
    validate_edges(edges, |i, edge| {
        (edge.source() == edge.target())
            .then(|| format!("edge {i}: self-loop on node {}", edge.source()))
    })
}

/// Verifies that all MST edge weights are finite.
fn validate_finite_weights(edges: &[MstEdge]) -> TestCaseResult {
    validate_edges(edges, |i, edge| {
        (!edge.weight().is_finite())
            .then(|| format!("edge {i}: non-finite weight {}", edge.weight()))
    })
}

/// Detects cycles in the MST output using union-find.
fn validate_acyclicity(node_count: usize, edges: &[MstEdge]) -> TestCaseResult {
    let mut parent: Vec<usize> = (0..node_count).collect();
    for (i, edge) in edges.iter().enumerate() {
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
    Ok(())
}

/// Verifies that the forest has exactly `n - c` edges for `c` components.
fn validate_edge_count(node_count: usize, actual: usize, component_count: usize) -> TestCaseResult {
    let expected = node_count.saturating_sub(component_count);
    if actual != expected {
        return Err(TestCaseError::fail(format!(
            "edge count {actual}, expected n - c = {expected} (n={node_count}, c={component_count})",
        )));
    }
    Ok(())
}

/// Verifies that a connected input produces a spanning tree.
fn validate_connectivity(
    fixture: &MstFixture,
    result: &crate::MinimumSpanningForest,
) -> TestCaseResult {
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
        if is_invalid_edge(s, t, n, edge.distance()) {
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
