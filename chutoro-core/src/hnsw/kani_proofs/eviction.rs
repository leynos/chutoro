//! Eviction-focused Kani harnesses for HNSW commit reconciliation.

use super::{EdgeAssertion, has_node_link};
use crate::hnsw::{
    error::HnswError,
    graph::{EdgeContext, Graph, NodeContext},
    insert::{
        FinalisedUpdate, NewNodeContext, StagedUpdate, apply_commit_updates_for_kani,
        test_helpers::add_edge_if_missing,
    },
    invariants::is_bidirectional,
    params::HnswParams,
};

/// Sets up a graph with 4 nodes at level 1 for eviction testing.
///
/// Returns a graph with nodes 0, 1, 2, 3 all inserted at level 1,
/// configured with `max_connections = 1` so that level 1 has capacity 1.
fn setup_eviction_test_graph(params: HnswParams) -> Result<Graph, HnswError> {
    let mut graph = Graph::with_capacity(params, 4);

    // Insert 4 nodes at level 1
    graph.insert_first(NodeContext {
        node: 0,
        level: 1,
        sequence: 0,
    })?;
    graph.attach_node(NodeContext {
        node: 1,
        level: 1,
        sequence: 1,
    })?;
    graph.attach_node(NodeContext {
        node: 2,
        level: 1,
        sequence: 2,
    })?;
    graph.attach_node(NodeContext {
        node: 3,
        level: 1,
        sequence: 3,
    })?;

    Ok(graph)
}

/// Verifies that eviction triggers correct deferred scrub behaviour.
///
/// This harness exercises the eviction path in `ensure_reverse_edge` and
/// verifies that `apply_deferred_scrubs` correctly removes orphaned forward
/// edges while maintaining the bidirectional invariant.
///
/// # Verification Bounds
///
/// - **Nodes**: 4 (IDs 0, 1, 2, 3)
/// - **Levels**: 2 (levels 0 and 1) to allow capacity-1 eviction on level 1
/// - **Edges**: Deterministic setup to trigger eviction and deferred scrub
///
/// # Scenario
///
/// 1. Node 1 is seeded at capacity (1 edge) with node 2 at level 1
/// 2. Node 0 adds node 1 as a neighbour at level 1
/// 3. `ensure_reverse_edge(origin=0, target=1)` evicts node 2 from node 1
/// 4. A `DeferredScrub { origin: 2, target: 1, level: 1 }` is created
/// 5. `apply_deferred_scrubs` removes the orphaned edge 2 → 1
///
/// # What This Proves
///
/// If this harness passes, Kani has verified that:
/// - Eviction correctly removes the furthest neighbour
/// - Deferred scrubs correctly remove orphaned forward edges
/// - The bidirectional invariant is maintained throughout
#[kani::proof]
#[kani::unwind(10)]
fn verify_eviction_deferred_scrub_reciprocity() {
    // Use max_connections = 1 so level 1 has capacity 1 and can evict.
    let Ok(params) = HnswParams::new(1, 2) else {
        kani::assert(false, "failed to construct eviction HNSW params");
        return;
    };
    let max_connections = params.max_connections();
    let setup_result = setup_eviction_test_graph(params);
    kani::assert(
        setup_result.is_ok(),
        "failed to construct eviction test graph",
    );
    let Ok(mut graph) = setup_result else {
        return;
    };

    // Seed node 1 at capacity with node 2 (bidirectional at level 1).
    // This ensures node 1's level-1 neighbour list is full.
    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    // Update: node 0 adds node 1 as neighbour at level 1.
    // When ensure_reverse_edge(origin=0, target=1) runs, node 1 is at
    // capacity, so node 2 is evicted and a deferred scrub is created.
    let update_ctx = EdgeContext {
        level: 1,
        max_connections,
    };
    let staged = StagedUpdate {
        node: 0,
        ctx: update_ctx,
        candidates: vec![1],
    };
    let updates: Vec<FinalisedUpdate> = vec![(staged, vec![1])];
    let new_node = NewNodeContext { id: 3, level: 1 };

    let commit_result =
        apply_commit_updates_for_kani(&mut graph, max_connections, new_node, updates);
    kani::assert(commit_result.is_ok(), "commit-path updates must succeed");
    if commit_result.is_err() {
        return;
    }

    // Assert bidirectional invariant holds after eviction and deferred scrub.
    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after eviction and deferred scrub",
    );

    // Assert node 1 links to node 0 (the new edge).
    kani::assert(
        has_node_link(&graph, EdgeAssertion::new(1, 0, 1)),
        "node 1 should link to node 0 after eviction",
    );

    // Assert node 2's forward edge to node 1 was scrubbed.
    kani::assert(
        !has_node_link(&graph, EdgeAssertion::new(2, 1, 1)),
        "deferred scrub should remove node 2's forward edge to node 1",
    );

    // Assert node 1 no longer links to node 2 (it was evicted).
    kani::assert(
        !has_node_link(&graph, EdgeAssertion::new(1, 2, 1)),
        "node 1 should no longer link to node 2 after eviction",
    );
}
