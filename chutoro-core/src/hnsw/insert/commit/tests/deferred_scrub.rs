//! Commit-path tests for deferred scrub behaviour.

use super::*;

/// Tests that a deferred scrub is benign when the targeted forward edge has
/// already been removed by an earlier operation.
///
/// Scenario:
/// - Node 1 is at capacity with node 2 (bidirectional edge 1↔2).
/// - Update 1: node 0 adds node 1, evicting node 2 from node 1. This queues
///   a deferred scrub for the orphaned (2 → 1) forward edge.
/// - Update 2: node 2 updates its neighbour list to include node 3 instead
///   of node 1. This removes the (2 → 1) edge before the deferred scrub runs.
/// - When deferred scrubs are applied, the scrub for (2 → 1) finds the edge
///   already absent and safely becomes a no-op.
///
/// Assertions:
/// - The test completes without panicking.
/// - Node 0 ↔ Node 1 edge exists (bidirectional).
/// - Node 2 → Node 1 edge is absent.
/// - Node 1 → Node 2 edge is absent.
/// - Node 2 ↔ Node 3 edge exists (from update 2).
#[rstest]
fn benign_deferred_scrub_is_noop_when_edge_already_removed(
    params_one_connection: HnswParams,
) -> Result<(), HnswError> {
    let max_connections = params_one_connection.max_connections();
    let mut graph = Graph::with_capacity(params_one_connection, 5);

    // Insert 5 nodes at level 1
    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;
    insert_node(&mut graph, 4, 1, 4)?;

    // Node 1 at capacity with node 2 (bidirectional)
    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    // Update 1: node 0 adds node 1 as neighbour.
    // This evicts node 2 from node 1's list and queues a deferred scrub for 2→1.
    let update1 = build_update(0, 1, vec![1], max_connections);

    // Update 2: node 2 replaces node 1 with node 3 in its neighbour list.
    // This removes edge 2→1 before the deferred scrub runs.
    let update2 = build_update(2, 1, vec![3], max_connections);

    let new_node = NewNodeContext { id: 4, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update1, update2], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    // The deferred scrub for 2→1 should be a no-op since update2 already
    // removed that edge. The test passing without panic confirms this.

    // Node 0 and node 1 should be linked (from update1)
    assert_bidirectional_edge(&graph, 0, 1, 1);

    // Node 2's forward edge to node 1 should be absent
    assert_no_edge(&graph, 2, 1, 1);

    // Node 1's forward edge to node 2 should be absent (evicted)
    assert_no_edge(&graph, 1, 2, 1);

    // Node 2 and node 3 should be linked (from update2)
    assert_bidirectional_edge(&graph, 2, 3, 1);

    Ok(())
}

/// Tests that deferred scrub is skipped when reciprocity is restored by a later
/// update.
///
/// Scenario: Node 1 is at capacity with node 2. First update evicts node 2, but
/// a second update re-adds the reciprocal edge. The deferred scrub should
/// detect the restored reciprocity and skip removing node 2's forward edge.
#[rstest]
fn eviction_skips_scrub_if_reciprocity_restored(
    params_one_connection: HnswParams,
) -> Result<(), HnswError> {
    let ctx = EvictionTestContext::new(params_one_connection)?;

    // First update: node 0 adds node 1 (evicts node 2 from node 1)
    // Second update: node 2 re-adds node 1 (restores the reciprocal edge)
    let update1 = build_update(0, 1, vec![1], ctx.max_connections);
    let update2 = build_update(2, 1, vec![1], ctx.max_connections);
    let graph = ctx.apply_updates(vec![update1, update2])?;

    // Node 2 should still have its forward edge to node 1 (scrub was skipped)
    // because the second update re-added node 1 to node 2's neighbour list
    assert_has_edge(&graph, 2, 1, 1);

    Ok(())
}

/// Tests that multiple evictions in a batch update are all scrubbed correctly.
///
/// Scenario: Two separate updates from different origin nodes each trigger an
/// eviction. Both orphaned forward edges should be scrubbed.
#[rstest]
fn multiple_evictions_in_batch_update(params_one_connection: HnswParams) -> Result<(), HnswError> {
    let max_connections = params_one_connection.max_connections();
    let mut graph = Graph::with_capacity(params_one_connection, 7);

    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;
    insert_node(&mut graph, 4, 1, 4)?;
    insert_node(&mut graph, 5, 1, 5)?;
    insert_node(&mut graph, 6, 1, 6)?;

    // Node 1 at capacity with node 2 (bidirectional)
    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    // Node 3 at capacity with node 4 (bidirectional)
    add_edge_if_missing(&mut graph, 3, 4, 1);
    add_edge_if_missing(&mut graph, 4, 3, 1);

    // First update: node 0 adds node 1 (evicts node 2 from node 1)
    // Second update: node 5 adds node 3 (evicts node 4 from node 3)
    // Using different origin nodes so each can succeed independently
    let update1 = build_update(0, 1, vec![1], max_connections);
    let update2 = build_update(5, 1, vec![3], max_connections);
    let new_node = NewNodeContext { id: 6, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update1, update2], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    // Both evicted nodes' forward edges should be scrubbed
    assert_no_edge(&graph, 2, 1, 1);
    assert_no_edge(&graph, 4, 3, 1);

    // The new edges should be bidirectional
    assert_bidirectional_edge(&graph, 0, 1, 1);
    assert_bidirectional_edge(&graph, 5, 3, 1);

    Ok(())
}

/// Tests that eviction respects furthest-first ordering.
///
/// Scenario: Node 1 has two neighbours (node 2 at front, node 3 at back).
/// When a new edge is added, the front entry (node 2, furthest) should be
/// evicted, not the back entry (node 3, closer).
#[rstest]
fn eviction_respects_furthest_first_ordering() -> Result<(), HnswError> {
    // Use max_connections = 2 so level-1 capacity is 2
    let params = HnswParams::new(2, 4)?;
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 5);

    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;
    insert_node(&mut graph, 4, 1, 4)?;

    // Seed node 1 at capacity with nodes 2 (furthest, front) and 3 (closer, back)
    // Order matters: push node 2 first (furthest), then node 3 (closer)
    let node1 = graph.node_mut(1).expect("node 1 should exist");
    node1.neighbours_mut(1).push(2); // furthest (front)
    node1.neighbours_mut(1).push(3); // closer (back)

    // Add reciprocal edges
    add_edge_if_missing(&mut graph, 2, 1, 1);
    add_edge_if_missing(&mut graph, 3, 1, 1);

    // Node 0 adds node 1, triggering eviction
    let update = build_update(0, 1, vec![1], max_connections);
    let new_node = NewNodeContext { id: 4, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    // Node 2 (furthest, front) should be evicted
    assert_no_edge(&graph, 1, 2, 1);
    assert_no_edge(&graph, 2, 1, 1);

    // Node 3 (closer, back) should remain
    assert_has_edge(&graph, 1, 3, 1);

    // New edge should be added
    assert_bidirectional_edge(&graph, 0, 1, 1);

    Ok(())
}

/// Context for base layer healing tests with a 4-node graph at level 0.
struct HealingTestContext {
    graph: Graph,
    max_connections: usize,
}

impl HealingTestContext {
    /// Creates a test graph with 4 nodes at level 0, where node 1 is at capacity
    /// with bidirectional edges to nodes 0 and 2, and node 2 is only connected
    /// to node 1 (will become isolated on eviction).
    fn new(params: HnswParams) -> Result<Self, HnswError> {
        let max_connections = params.max_connections();
        let mut graph = Graph::with_capacity(params, 4);

        // All nodes at level 0 (base layer)
        insert_node(&mut graph, 0, 0, 0)?; // entry node
        insert_node(&mut graph, 1, 0, 1)?;
        insert_node(&mut graph, 2, 0, 2)?;
        insert_node(&mut graph, 3, 0, 3)?;

        // Node 1 at capacity (level 0 has 2*max_connections = 2) with nodes 2 and 0
        add_edge_if_missing(&mut graph, 1, 2, 0);
        add_edge_if_missing(&mut graph, 1, 0, 0);
        add_edge_if_missing(&mut graph, 2, 1, 0);
        add_edge_if_missing(&mut graph, 0, 1, 0);

        // Node 2 only connected to node 1
        // After eviction, node 2 becomes isolated

        Ok(Self {
            graph,
            max_connections,
        })
    }

    /// Applies the given updates and returns the graph for assertions.
    fn apply_updates(
        mut self,
        updates: Vec<(StagedUpdate, Vec<usize>)>,
        new_node: NewNodeContext,
    ) -> Result<Graph, HnswError> {
        let mut applicator = CommitApplicator::new(&mut self.graph);
        let (reciprocated, _) =
            applicator.apply_neighbour_updates(updates, self.max_connections, new_node)?;
        applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;
        Ok(self.graph)
    }
}

/// Tests that eviction at the base layer triggers connectivity healing.
///
/// Scenario: At level 0, when a node becomes isolated due to eviction,
/// the connectivity healer should restore a direct link to the entry node
/// (node 0) at the base layer, ensuring that connectivity is explicitly
/// maintained via the entry point.
#[rstest]
fn eviction_at_base_layer_triggers_healing() -> Result<(), HnswError> {
    let params = HnswParams::new(1, 4)?;
    let ctx = HealingTestContext::new(params)?;

    // Node 3 adds node 1, triggering eviction of node 2 from node 1
    let update = build_update(3, 0, vec![1], ctx.max_connections);
    let new_node = NewNodeContext { id: 3, level: 0 };

    let graph = ctx.apply_updates(vec![update], new_node)?;

    // Node 2 should have been healed to connect to the entry node (node 0)
    let node2 = graph.node(2).expect("node 2 should exist");
    assert!(
        node2.neighbours(0).contains(&0),
        "node 2 should be healed to connect to entry node 0",
    );

    Ok(())
}
