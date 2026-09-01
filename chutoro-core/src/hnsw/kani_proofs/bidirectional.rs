//! Bidirectional-link Kani harnesses for bounded HNSW graphs.

use super::add_bidirectional_edge;
use crate::hnsw::{
    graph::{Graph, NodeContext},
    insert::{KaniUpdateContext, ensure_reverse_edge_for_kani, test_helpers::add_edge_if_missing},
    invariants::is_bidirectional,
    params::HnswParams,
};

/// Smoke-checks that a tiny symmetric graph satisfies the invariant.
///
/// This harness is deterministic and intended to validate the Kani
/// toolchain wiring with minimal solver work.
#[kani::proof]
#[kani::unwind(4)]
fn verify_bidirectional_links_smoke_2_nodes_1_layer() {
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
        kani::assert(false, "Kani smoke insert must succeed");
        return;
    };
    let Ok(()) = graph.attach_node_for_kani(NodeContext {
        node: 1,
        level: 0,
        sequence: 1,
    }) else {
        kani::assert(false, "Kani smoke attach must succeed");
        return;
    };

    add_bidirectional_edge(&mut graph, 0, 1, 0);

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated in smoke harness",
    );
}

/// Verifies that reconciliation preserves bidirectional links.
///
/// This harness exercises the production reconciliation path used during
/// insertion commit. It applies a nondeterministic forward edge and then
/// invokes `EdgeReconciler::ensure_reverse_edge` via
/// `ensure_reverse_edge_for_kani` to enforce reciprocity.
///
/// # Verification Bounds
///
/// - **Nodes**: 2 (IDs 0, 1)
/// - **Layers**: 1 (base layer only, level 0)
/// - **Updates**: Nondeterministic neighbour list for node 0
#[kani::proof]
#[kani::unwind(4)]
fn verify_bidirectional_links_reconciliation_2_nodes_1_layer() {
    let Ok(params) = HnswParams::new_for_kani(1, 1) else {
        kani::assert(false, "Kani params must be valid");
        return;
    };
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 2);

    let Ok(()) = graph.insert_first_for_kani(NodeContext {
        node: 0,
        level: 0,
        sequence: 0,
    }) else {
        kani::assert(false, "Kani reconciliation insert must succeed");
        return;
    };
    let Ok(()) = graph.attach_node_for_kani(NodeContext {
        node: 1,
        level: 0,
        sequence: 1,
    }) else {
        kani::assert(false, "Kani reconciliation attach must succeed");
        return;
    };
    let should_link = kani::any::<bool>();
    if should_link {
        add_edge_if_missing(&mut graph, 0, 1, 0);
        let ctx = KaniUpdateContext::new(0, 0, max_connections);
        let added = ensure_reverse_edge_for_kani(&mut graph, ctx, 1);
        kani::assert(added, "expected reverse edge to be inserted");
    }

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after reconciliation",
    );
}
