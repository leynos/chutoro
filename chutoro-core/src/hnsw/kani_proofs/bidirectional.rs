//! Bidirectional-link Kani harnesses for bounded HNSW graphs.

use super::add_bidirectional_edge;
use crate::hnsw::{
    graph::{EdgeContext, Graph, NodeContext},
    insert::{
        FinalisedUpdate, KaniUpdateContext, NewNodeContext, StagedUpdate,
        apply_commit_updates_for_kani, apply_reconciled_update_for_kani,
        ensure_reverse_edge_for_kani, test_helpers::add_edge_if_missing,
    },
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

/// Builds the 3-node, 2-level graph used by the commit-path harness.
///
/// Inserts nodes 0, 1, and 2 at level 1 and seeds a bidirectional edge
/// between nodes 0 and 2 so that node 0's level-1 neighbour list is at
/// capacity before the commit path runs.
///
/// Returns `(graph, max_connections)` on success.
fn setup_commit_path_graph() -> Result<(Graph, usize), &'static str> {
    let params = HnswParams::new_for_kani(1, 2)?;
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 3);
    graph.insert_first_for_kani(NodeContext {
        node: 0,
        level: 1,
        sequence: 0,
    })?;
    graph.attach_node_for_kani(NodeContext {
        node: 1,
        level: 1,
        sequence: 1,
    })?;
    graph.attach_node_for_kani(NodeContext {
        node: 2,
        level: 1,
        sequence: 2,
    })?;
    add_edge_if_missing(&mut graph, 0, 2, 1);
    add_edge_if_missing(&mut graph, 2, 0, 1);
    Ok((graph, max_connections))
}

/// Verifies that HNSW graph edges are bidirectional (symmetric).
///
/// This harness drives the production commit-path reconciliation logic to
/// ensure that bidirectional edges and deferred scrubs produce a symmetric
/// graph for a bounded configuration.
///
/// # Verification Bounds
///
/// - **Nodes**: 3 (IDs 0, 1, 2)
/// - **Levels**: 2 (levels 0 and 1) to allow capacity-1 eviction on level 1
/// - **Edges**: Deterministic setup to trigger a deferred scrub
///
/// # Invariant Under Test
///
/// The bidirectional links invariant states that for every directed edge
/// `(u, v)` at level `l`, there must exist a reverse edge `(v, u)` at the
/// same level. This is essential for HNSW search correctness.
///
/// # What This Proves
///
/// If this harness passes, Kani has verified that the commit-path
/// reconciliation logic (including deferred scrubs) produces a bidirectional
/// graph for the bounded configuration.
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(10)]
fn verify_bidirectional_links_commit_path_3_nodes() {
    let Ok((mut graph, max_connections)) = setup_commit_path_graph() else {
        kani::assert(false, "commit-path graph setup must succeed");
        return;
    };

    let Some(node_zero) = graph.node(0) else {
        kani::assert(false, "node 0 must exist after seeding commit-path edge");
        return;
    };
    let Some(node_two) = graph.node(2) else {
        kani::assert(false, "node 2 must exist after seeding commit-path edge");
        return;
    };
    kani::assert(
        node_zero.neighbours(1).contains(&2),
        "node 0 must contain seeded level-1 edge to node 2",
    );
    kani::assert(
        node_two.neighbours(1).contains(&0),
        "node 2 must contain seeded level-1 edge to node 0",
    );

    let update_ctx = EdgeContext {
        level: 1,
        max_connections,
    };
    let staged = StagedUpdate {
        node: 1,
        ctx: update_ctx,
        candidates: vec![0],
    };
    let updates: Vec<FinalisedUpdate> = vec![(staged, vec![0])];
    let new_node = NewNodeContext { id: 1, level: 1 };
    let Ok(()) = apply_commit_updates_for_kani(&mut graph, max_connections, new_node, updates)
    else {
        kani::assert(false, "commit-path updates must succeed");
        return;
    };

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after commit-path reconciliation",
    );

    let Some(node_two) = graph.node(2) else {
        kani::assert(false, "node 2 must exist after commit-path reconciliation");
        return;
    };
    let node_two_has_edge = node_two.neighbours(1).contains(&0);
    kani::assert(
        !node_two_has_edge,
        "deferred scrub should remove evicted forward edge",
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

/// Verifies reconciliation on a 3-node graph (heavier, but broader coverage).
///
/// This harness is intentionally more expensive and is intended for
/// `make kani-full` runs rather than the default `make kani`.
#[kani::proof]
#[kani::solver(kissat)]
#[kani::unwind(10)]
fn verify_bidirectional_links_reconciliation_3_nodes_1_layer() {
    let Ok(params) = HnswParams::new_for_kani(2, 2) else {
        kani::assert(false, "Kani params must be valid");
        return;
    };
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 3);

    let Ok(()) = graph.insert_first_for_kani(NodeContext {
        node: 0,
        level: 0,
        sequence: 0,
    }) else {
        kani::assert(false, "failed to insert node 0");
        return;
    };
    let Ok(()) = graph.attach_node_for_kani(NodeContext {
        node: 1,
        level: 0,
        sequence: 1,
    }) else {
        kani::assert(false, "failed to attach node 1");
        return;
    };
    let Ok(()) = graph.attach_node_for_kani(NodeContext {
        node: 2,
        level: 0,
        sequence: 2,
    }) else {
        kani::assert(false, "failed to attach node 2");
        return;
    };

    // Exercise a replacement transition: reconciliation must remove the
    // reciprocal edge to node 1 and add the reciprocal edge to node 2.
    add_bidirectional_edge(&mut graph, 0, 1, 0);
    let mut next = vec![2];

    let ctx = KaniUpdateContext::new(0, 0, max_connections);
    apply_reconciled_update_for_kani(&mut graph, ctx, &mut next);

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after reconciliation",
    );
}
