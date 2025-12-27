//! Kani formal verification harnesses for HNSW graph invariants.
//!
//! These harnesses use bounded model checking to exhaustively verify that
//! structural invariants hold for all possible graph configurations within
//! the specified bounds. Kani explores every possible combination of
//! nondeterministic choices, providing formal guarantees rather than
//! probabilistic coverage.
//!
//! # Running Harnesses
//!
//! ```bash
//! cargo kani -p chutoro-core --harness verify_bidirectional_links_3_nodes_1_layer
//! ```
//!
//! Or via the Makefile:
//!
//! ```bash
//! make kani
//! ```
//!
//! # Relationship to Property Testing
//!
//! These harnesses complement the proptest-based property tests in
//! [`crate::hnsw::tests::property`]. While proptest provides probabilistic
//! coverage over large input spaces, Kani provides exhaustive coverage over
//! small, bounded configurations. Together they form a comprehensive
//! verification strategy.

use crate::hnsw::{
    graph::{Graph, NodeContext},
    invariants::is_bidirectional,
    insert::apply_reconciled_update_for_kani,
    params::HnswParams,
};

/// Verifies that HNSW graph edges are bidirectional (symmetric).
///
/// For a bounded graph with 3 nodes and 1 layer, exhaustively checks that
/// every edge `(source, target)` has a corresponding reverse edge
/// `(target, source)` at the same layer.
///
/// # Verification Bounds
///
/// - **Nodes**: 3 (IDs 0, 1, 2)
/// - **Layers**: 1 (base layer only, level 0)
/// - **Edges**: Nondeterministically populated via `kani::any()`
///
/// # Invariant Under Test
///
/// The bidirectional links invariant states that for every directed edge
/// `(u, v)` at layer `l`, there must exist a reverse edge `(v, u)` at the
/// same layer. This is essential for HNSW search correctness.
///
/// # What This Proves
///
/// If this harness passes, Kani has verified that for all 2^6 = 64 possible
/// edge configurations of a 3-node graph, enforcing bidirectionality via
/// [`enforce_bidirectional_constraint`] produces a graph where the invariant
/// holds.
#[kani::proof]
#[kani::unwind(10)]
fn verify_bidirectional_links_3_nodes_1_layer() {
    // Create minimal valid HNSW parameters (M=2, ef_construction=2)
    let params = HnswParams::new(2, 2).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 3);

    // Insert 3 nodes at level 0
    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert node 0");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach node 1");
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .expect("attach node 2");

    // Nondeterministically populate edges - Kani explores all combinations
    populate_edges_nondeterministically(&mut graph);

    // Enforce bidirectionality: for each edge, ensure reverse exists
    enforce_bidirectional_constraint(&mut graph);

    // Assert the bidirectional invariant holds for the final graph state
    // Uses the shared helper from invariants module to ensure alignment
    // with the production invariant definition.
    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated: missing reverse edge",
    );
}

/// Verifies that reconciliation preserves bidirectional links.
///
/// This harness exercises the production reconciliation path used during
/// insertion commit. It starts from a bidirectional baseline graph, applies a
/// nondeterministic update to a single node, and then invokes the real
/// reconciliation logic to enforce reciprocity.
///
/// # Verification Bounds
///
/// - **Nodes**: 3 (IDs 0, 1, 2)
/// - **Layers**: 1 (base layer only, level 0)
/// - **Updates**: Nondeterministic neighbour list for node 0
#[kani::proof]
#[kani::unwind(10)]
fn verify_bidirectional_links_reconciliation_3_nodes_1_layer() {
    let params = HnswParams::new(2, 2).expect("params must be valid");
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 3);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert node 0");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach node 1");
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .expect("attach node 2");

    // Seed a bidirectional baseline graph.
    if kani::any::<bool>() {
        add_bidirectional_edge(&mut graph, 0, 1, 0);
    }
    if kani::any::<bool>() {
        add_bidirectional_edge(&mut graph, 0, 2, 0);
    }
    if kani::any::<bool>() {
        add_bidirectional_edge(&mut graph, 1, 2, 0);
    }

    // Proposed trimmed neighbours for node 0 (may add or remove edges).
    let mut next: Vec<usize> = Vec::new();
    if kani::any::<bool>() {
        push_if_absent(&mut next, 1);
    }
    if kani::any::<bool>() {
        push_if_absent(&mut next, 2);
    }

    apply_reconciled_update_for_kani(&mut graph, 0, 0, max_connections, &mut next);

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after reconciliation",
    );
}

/// Populates edges nondeterministically using Kani's symbolic execution.
///
/// For a 3-node graph, there are 6 potential directed edges (excluding
/// self-loops). Each edge is independently decided by `kani::any()`,
/// resulting in 2^6 = 64 possible edge configurations to verify.
fn populate_edges_nondeterministically(graph: &mut Graph) {
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];

    // Explicit degree bound matching the HNSW M parameter (max_connections).
    // This helps Kani's bounded model checker by constraining Vec growth.
    const MAX_DEGREE: usize = 2;

    for (source, target) in EDGES {
        // Skip this edge if Kani decides not to include it
        if !kani::any::<bool>() {
            continue;
        }

        // Get mutable reference to source node, skip if not found
        let Some(node) = graph.node_mut(source) else {
            continue;
        };

        let neighbours = node.neighbours_mut(0);

        // Skip if already at max degree (bounds state space for Kani)
        if neighbours.len() >= MAX_DEGREE {
            continue;
        }

        // Only add if not already present
        if !neighbours.contains(&target) {
            neighbours.push(target);
        }
    }
}

/// Enforces bidirectional constraint: for each edge, ensure reverse exists.
///
/// This simulates the reciprocity enforcement that the HNSW insertion
/// algorithm performs. After this function, every edge should have a
/// corresponding reverse edge.
fn enforce_bidirectional_constraint(graph: &mut Graph) {
    // Collect edges first to avoid borrow conflicts
    let mut edges_to_add: Vec<(usize, usize)> = Vec::new();

    for (source, node) in graph.nodes_iter() {
        for &target in node.neighbours(0) {
            edges_to_add.push((target, source));
        }
    }

    // Add reverse edges where missing
    for (source, target) in edges_to_add {
        add_reverse_edge_if_missing(graph, source, target);
    }
}

/// Adds a reverse edge from `source` to `target` if it doesn't already exist.
///
/// This helper abstracts the pattern of conditionally adding an edge,
/// reducing nesting in the calling code. The degree bound is intentionally
/// not enforced here since reciprocity enforcement may temporarily exceed
/// the M parameter before trimming occurs in production code.
fn add_reverse_edge_if_missing(graph: &mut Graph, source: usize, target: usize) {
    let Some(node) = graph.node_mut(source) else {
        return;
    };

    let neighbours = node.neighbours_mut(0);
    if !neighbours.contains(&target) {
        neighbours.push(target);
    }
}

fn add_bidirectional_edge(graph: &mut Graph, origin: usize, target: usize, level: usize) {
    add_edge_if_missing(graph, origin, target, level);
    add_edge_if_missing(graph, target, origin, level);
}

fn add_edge_if_missing(graph: &mut Graph, origin: usize, target: usize, level: usize) {
    let Some(node) = graph.node_mut(origin) else {
        return;
    };
    let neighbours = node.neighbours_mut(level);
    if !neighbours.contains(&target) {
        neighbours.push(target);
    }
}

fn push_if_absent(list: &mut Vec<usize>, value: usize) {
    if !list.contains(&value) {
        list.push(value);
    }
}
