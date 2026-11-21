//! Kani harnesses for base-layer connectivity healing.
//!
//! `ConnectivityHealer::ensure_base_connectivity` repairs a node that insertion
//! reconciliation left without a base-layer link. The healer uses a work queue
//! and a visited set, so its behaviour is stateful; the harness drives the
//! unchanged production routine on a bounded graph and asserts the resulting
//! adjacency lists rather than re-implementing the algorithm.
//!
//! The retained configuration is a concrete two-node fixture. A three-node
//! probe that also added one symbolic boolean and raised the unwind bound
//! exceeded 600 seconds of solving with thousands of aborted paths, so the
//! proof stops at the smallest case that exercises link creation. That probe
//! changed three factors together and does not isolate which one dominated;
//! the wider behaviour stays with the normal healing tests.

use crate::hnsw::{
    graph::{Graph, NodeContext},
    insert::ensure_base_connectivity_for_kani,
    params::HnswParams,
};

/// Reports whether a node exposes level 0 with an empty base list.
fn has_empty_base_list(graph: &Graph, node: usize) -> bool {
    graph
        .node(node)
        .is_some_and(|node| node.level_count() > 0 && node.neighbours(0).is_empty())
}

/// Reports whether a node's base list is exactly `expected`.
fn base_list_is(graph: &Graph, node: usize, expected: &[usize]) -> bool {
    graph
        .node(node)
        .is_some_and(|node| node.neighbours(0) == expected)
}

/// Verifies that base-layer healing links an isolated node to the entry node.
///
/// The graph is two level-0 nodes with empty adjacency lists. Node 0 is the
/// entry point and node 1 is isolated. The production healer is asked to heal
/// node 1 with a base-layer connection limit of two, which has spare capacity
/// on both lists, so no eviction can occur. The exact postconditions are that
/// node 0 lists `[1]` and node 1 lists `[0]`; the entry point is unchanged.
///
/// # Verification Bounds
///
/// - **Nodes**: 2 (IDs 0, 1), both exposing only level 0
/// - **Levels**: 0 (base layer)
/// - **Edges**: none before the call; both lists empty
/// - **Connection limit**: 2 at the base layer (`max_connections = 1`)
/// - **Unwind**: 9, which covers the eight-byte `memcmp` loop used by the
///   exact one-element slice comparisons
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(9)]
fn verify_base_layer_healing_links_isolated_node_2_nodes() {
    let Ok(params) = HnswParams::new_for_kani(1, 1) else {
        kani::assert(false, "Kani params must be valid");
        return;
    };
    let mut graph = Graph::with_capacity(params, 2);

    let Ok(()) = graph.insert_first_for_kani(NodeContext {
        node: 0,
        level: 0,
        sequence: 0,
    }) else {
        kani::assert(false, "Kani healing insert must succeed");
        return;
    };
    let Ok(()) = graph.attach_node_for_kani(NodeContext {
        node: 1,
        level: 0,
        sequence: 1,
    }) else {
        kani::assert(false, "Kani healing attach must succeed");
        return;
    };

    kani::assert(
        has_empty_base_list(&graph, 0) && has_empty_base_list(&graph, 1),
        "healing precondition: both base lists must start empty",
    );
    kani::assert(
        graph.entry().is_some_and(|entry| entry.node == 0),
        "healing precondition: node 0 must be the entry",
    );
    kani::cover!(
        has_empty_base_list(&graph, 0) && has_empty_base_list(&graph, 1),
        "setup witness: both base lists are empty"
    );

    ensure_base_connectivity_for_kani(&mut graph, 1, 1);

    kani::assert(
        base_list_is(&graph, 0, &[1]),
        "healing must link the entry node to the isolated node",
    );
    kani::assert(
        base_list_is(&graph, 1, &[0]),
        "healing must link the isolated node back to the entry node",
    );
    kani::assert(
        graph.entry().is_some_and(|entry| entry.node == 0),
        "healing must preserve the entry point",
    );
    kani::cover!(
        base_list_is(&graph, 0, &[1]) && base_list_is(&graph, 1, &[0]),
        "healing witness: the reciprocal base-layer link exists"
    );
}
