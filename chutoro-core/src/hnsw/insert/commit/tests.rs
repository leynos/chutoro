//! Commit-path tests for neighbour updates and deferred scrubs.

use super::super::limits;
use super::super::test_helpers::{add_edge_if_missing, assert_no_edge};
use super::CommitApplicator;
use crate::hnsw::{
    error::HnswError,
    graph::{EdgeContext, Graph, NodeContext},
    insert::types::{NewNodeContext, StagedUpdate},
    params::HnswParams,
};
use rstest::{fixture, rstest};

#[fixture]
fn params_two_connections() -> HnswParams {
    HnswParams::new(2, 4).expect("params should be valid for tests")
}

fn insert_node(
    graph: &mut Graph,
    node: usize,
    level: usize,
    sequence: u64,
) -> Result<(), HnswError> {
    if graph.entry().is_none() {
        graph.insert_first(NodeContext {
            node,
            level,
            sequence,
        })?;
    } else {
        graph.attach_node(NodeContext {
            node,
            level,
            sequence,
        })?;
    }
    Ok(())
}

fn assert_bidirectional_edge(graph: &Graph, node_a: usize, node_b: usize, level: usize) {
    let a_msg = format!("node {node_a} should exist");
    let a = graph.node(node_a).expect(&a_msg);
    let b_msg = format!("node {node_b} should exist");
    let b = graph.node(node_b).expect(&b_msg);
    assert!(
        a.level_count() > level && b.level_count() > level,
        "both nodes must expose level {level}",
    );
    assert!(
        a.neighbours(level).contains(&node_b),
        "expected edge {node_a}->{node_b} at level {level}",
    );
    assert!(
        b.neighbours(level).contains(&node_a),
        "expected edge {node_b}->{node_a} at level {level}",
    );
}

fn build_update(
    node: usize,
    level: usize,
    neighbours: Vec<usize>,
    max_connections: usize,
) -> (StagedUpdate, Vec<usize>) {
    let ctx = EdgeContext {
        level,
        max_connections,
    };
    let staged = StagedUpdate {
        node,
        ctx,
        candidates: neighbours.clone(),
    };
    (staged, neighbours)
}

#[rstest]
#[case::base_layer(0)]
#[case::upper_layer(1)]
fn commit_updates_write_reciprocal_edges(
    #[case] level: usize,
    params_two_connections: HnswParams,
) -> Result<(), HnswError> {
    let max_connections = params_two_connections.max_connections();
    let mut graph = Graph::with_capacity(params_two_connections.clone(), 3);

    insert_node(&mut graph, 0, level, 0)?;
    insert_node(&mut graph, 1, level, 1)?;
    insert_node(&mut graph, 2, level, 2)?;

    add_edge_if_missing(&mut graph, 0, 1, level);
    add_edge_if_missing(&mut graph, 1, 0, level);

    let update = build_update(0, level, vec![1, 2], max_connections);
    let new_node = NewNodeContext { id: 2, level };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    assert_bidirectional_edge(&graph, 0, 2, level);
    assert_bidirectional_edge(&graph, 0, 1, level);

    Ok(())
}

#[rstest]
fn commit_updates_scrub_evicted_forward_edge() -> Result<(), HnswError> {
    let params = HnswParams::new(1, 4)?;
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 4);

    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;

    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    let update = build_update(0, 1, vec![1], max_connections);
    let new_node = NewNodeContext { id: 3, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    let limit = limits::compute_connection_limit(1, max_connections);
    for node_id in [0, 1, 2, 3] {
        let node_msg = format!("node {node_id} should exist");
        let node = graph.node(node_id).expect(&node_msg);
        assert!(
            node.neighbours(1).len() <= limit,
            "expected node {node_id} to respect level-1 connection limit",
        );
    }

    assert_bidirectional_edge(&graph, 0, 1, 1);
    assert_no_edge(&graph, 2, 1, 1);
    assert_no_edge(&graph, 1, 2, 1);

    Ok(())
}

#[rstest]
fn commit_updates_report_missing_origin(params_two_connections: HnswParams) {
    let max_connections = params_two_connections.max_connections();
    let mut graph = Graph::with_capacity(params_two_connections, 2);

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

    let update = build_update(99, 0, vec![0], max_connections);
    let new_node = NewNodeContext { id: 1, level: 0 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let err = applicator
        .apply_neighbour_updates(vec![update], max_connections, new_node)
        .expect_err("missing origin should error");

    assert!(matches!(err, HnswError::GraphInvariantViolation { .. }));
}

// ---------------------------------------------------------------------------
// Eviction and deferred scrub tests
// ---------------------------------------------------------------------------

#[fixture]
fn params_one_connection() -> HnswParams {
    HnswParams::new(1, 4).expect("params should be valid for tests")
}

fn assert_has_edge(graph: &Graph, origin: usize, target: usize, level: usize) {
    let node = graph.node(origin).expect("node should exist");
    assert!(
        level < node.level_count(),
        "node {origin} should expose level {level}",
    );
    assert!(
        node.neighbours(level).contains(&target),
        "expected edge {origin}->{target} at level {level}",
    );
}

/// Tests that eviction correctly scrubs orphaned forward edges.
///
/// Scenario: Node 1 is at capacity with node 2. When node 0 adds node 1 as a
/// neighbour, node 2 is evicted from node 1's neighbour list. The deferred
/// scrub should then remove node 2's forward edge to node 1.
#[rstest]
fn eviction_scrubs_orphaned_forward_edge(
    params_one_connection: HnswParams,
) -> Result<(), HnswError> {
    let max_connections = params_one_connection.max_connections();
    let mut graph = Graph::with_capacity(params_one_connection, 4);

    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;

    // Seed node 1 at capacity with node 2 (bidirectional)
    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    // Node 0 adds node 1 as neighbour, triggering eviction of node 2
    let update = build_update(0, 1, vec![1], max_connections);
    let new_node = NewNodeContext { id: 3, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    // Node 0 and node 1 should be linked
    assert_bidirectional_edge(&graph, 0, 1, 1);

    // Node 2's forward edge to node 1 should be scrubbed
    assert_no_edge(&graph, 2, 1, 1);

    // Node 1 should no longer link to node 2
    assert_no_edge(&graph, 1, 2, 1);

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
    let max_connections = params_one_connection.max_connections();
    let mut graph = Graph::with_capacity(params_one_connection, 4);

    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;

    // Seed node 1 at capacity with node 2 (bidirectional)
    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    // First update: node 0 adds node 1 (evicts node 2 from node 1)
    // Second update: node 2 re-adds node 1 (restores the reciprocal edge)
    let update1 = build_update(0, 1, vec![1], max_connections);
    let update2 = build_update(2, 1, vec![1], max_connections);
    let new_node = NewNodeContext { id: 3, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update1, update2], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

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

/// Tests that eviction at the base layer triggers connectivity healing.
///
/// Scenario: At level 0, when a node becomes isolated due to eviction,
/// the connectivity healer should restore a link to the entry node.
#[rstest]
fn eviction_at_base_layer_triggers_healing() -> Result<(), HnswError> {
    let params = HnswParams::new(1, 4)?;
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

    // Node 3 adds node 1, triggering eviction of node 2 from node 1
    let update = build_update(3, 0, vec![1], max_connections);
    let new_node = NewNodeContext { id: 3, level: 0 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    // Node 2 should have been healed to connect to the entry node (node 0)
    // or some other reachable node
    let node2 = graph.node(2).expect("node 2 should exist");
    assert!(
        !node2.neighbours(0).is_empty(),
        "node 2 should not be isolated after healing",
    );

    Ok(())
}
