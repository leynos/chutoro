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
//! cargo kani -p chutoro-core --harness verify_bidirectional_links_commit_path_3_nodes
//! ```
//!
//! Or via the Makefile (practical harnesses):
//!
//! ```bash
//! make kani
//! ```
//!
//! Run the full suite (includes heavier 3-node harnesses):
//!
//! ```bash
//! make kani-full
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
    let params = HnswParams::new(1, 1).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 2);

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

    add_bidirectional_edge(&mut graph, 0, 1, 0);

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated in smoke harness",
    );
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
#[kani::unwind(10)]
fn verify_bidirectional_links_commit_path_3_nodes() {
    // Use max_connections = 1 so level 1 has capacity 1 and can evict.
    let params = HnswParams::new(1, 2).expect("params must be valid");
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 3);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 1,
            sequence: 0,
        })
        .expect("insert node 0");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 1,
            sequence: 1,
        })
        .expect("attach node 1");
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 1,
            sequence: 2,
        })
        .expect("attach node 2");

    // Seed node 0's level-1 neighbour list so it is at capacity with node 2.
    add_edge_if_missing(&mut graph, 0, 2, 1);
    add_edge_if_missing(&mut graph, 2, 0, 1);

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
    apply_commit_updates_for_kani(&mut graph, max_connections, new_node, updates)
        .expect("commit-path updates must succeed");

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after commit-path reconciliation",
    );

    let node_two_has_edge = graph
        .node(2)
        .map(|node| node.neighbours(1).contains(&0))
        .unwrap_or(false);
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
    let params = HnswParams::new(1, 1).expect("params must be valid");
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 2);

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

    let ctx = KaniUpdateContext::new(0, 0, max_connections);
    apply_reconciled_update_for_kani(&mut graph, ctx, &mut next);

    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after reconciliation",
    );
}

/// Sets up a graph with 4 nodes at level 1 for eviction testing.
///
/// Returns a graph with nodes 0, 1, 2, 3 all inserted at level 1,
/// configured with `max_connections = 1` so that level 1 has capacity 1.
#[cfg(kani)]
fn setup_eviction_test_graph(params: HnswParams) -> Graph {
    let mut graph = Graph::with_capacity(params, 4);

    // Insert 4 nodes at level 1
    graph
        .insert_first(NodeContext {
            node: 0,
            level: 1,
            sequence: 0,
        })
        .expect("insert node 0");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 1,
            sequence: 1,
        })
        .expect("attach node 1");
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 1,
            sequence: 2,
        })
        .expect("attach node 2");
    graph
        .attach_node(NodeContext {
            node: 3,
            level: 1,
            sequence: 3,
        })
        .expect("attach node 3");

    graph
}

#[cfg(kani)]
struct EdgeAssertion {
    source: usize,
    target: usize,
    level: usize,
}

#[cfg(kani)]
impl EdgeAssertion {
    fn new(source: usize, target: usize, level: usize) -> Self {
        Self {
            source,
            target,
            level,
        }
    }
}

/// Asserts that source links to target at the given level.
#[cfg(kani)]
fn assert_node_link(graph: &Graph, edge: EdgeAssertion, message: &str) {
    let has_link = graph
        .node(edge.source)
        .map(|n| n.neighbours(edge.level).contains(&edge.target))
        .unwrap_or(false);
    kani::assert(has_link, message);
}

/// Asserts that source does NOT link to target at the given level.
#[cfg(kani)]
fn assert_no_node_link(graph: &Graph, edge: EdgeAssertion, message: &str) {
    let has_link = graph
        .node(edge.source)
        .map(|n| n.neighbours(edge.level).contains(&edge.target))
        .unwrap_or(false);
    kani::assert(!has_link, message);
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
    let params = HnswParams::new(1, 2).expect("params must be valid");
    let max_connections = params.max_connections();
    let mut graph = setup_eviction_test_graph(params);

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

    apply_commit_updates_for_kani(&mut graph, max_connections, new_node, updates)
        .expect("commit-path updates must succeed");

    // Assert bidirectional invariant holds after eviction and deferred scrub.
    kani::assert(
        is_bidirectional(&graph),
        "bidirectional invariant violated after eviction and deferred scrub",
    );

    // Assert node 1 links to node 0 (the new edge).
    assert_node_link(
        &graph,
        EdgeAssertion::new(1, 0, 1),
        "node 1 should link to node 0 after eviction",
    );

    // Assert node 2's forward edge to node 1 was scrubbed.
    assert_no_node_link(
        &graph,
        EdgeAssertion::new(2, 1, 1),
        "deferred scrub should remove node 2's forward edge to node 1",
    );

    // Assert node 1 no longer links to node 2 (it was evicted).
    assert_no_node_link(
        &graph,
        EdgeAssertion::new(1, 2, 1),
        "node 1 should no longer link to node 2 after eviction",
    );
}

fn add_bidirectional_edge(graph: &mut Graph, origin: usize, target: usize, level: usize) {
    add_edge_if_missing(graph, origin, target, level);
    add_edge_if_missing(graph, target, origin, level);
}

fn push_if_absent(list: &mut Vec<usize>, value: usize) {
    if !list.contains(&value) {
        list.push(value);
    }
}
