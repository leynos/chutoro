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
    match HnswParams::new(2, 4) {
        Ok(params) => params,
        Err(err) => panic!("params should be valid for tests: {err}"),
    }
}

fn insert_node(
    graph: &mut Graph,
    node: usize,
    level: usize,
    sequence: u64,
) -> Result<(), HnswError> {
    let ctx = NodeContext {
        node,
        level,
        sequence,
    };
    if graph.entry().is_none() {
        graph.insert_first(ctx)
    } else {
        graph.attach_node(ctx)
    }
}

fn assert_bidirectional_edge(graph: &Graph, node_a: usize, node_b: usize, level: usize) {
    let Some(a) = graph.node(node_a) else {
        panic!("node {node_a} should exist");
    };
    let Some(b) = graph.node(node_b) else {
        panic!("node {node_b} should exist");
    };
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
    match HnswParams::new(1, 4) {
        Ok(params) => params,
        Err(err) => panic!("params should be valid for tests: {err}"),
    }
}

fn assert_has_edge(graph: &Graph, origin: usize, target: usize, level: usize) {
    let Some(node) = graph.node(origin) else {
        panic!("node {origin} should exist");
    };
    assert!(
        level < node.level_count(),
        "node {origin} should expose level {level}",
    );
    assert!(
        node.neighbours(level).contains(&target),
        "expected edge {origin}->{target} at level {level}",
    );
}

/// Context for eviction tests with a 4-node graph where node 1 is at capacity.
struct EvictionTestContext {
    graph: Graph,
    max_connections: usize,
    new_node: NewNodeContext,
}

impl EvictionTestContext {
    /// Creates a test graph with 4 nodes at level 1, where node 1 is seeded
    /// at capacity with a bidirectional edge to node 2.
    fn new(params: HnswParams) -> Result<Self, HnswError> {
        let max_connections = params.max_connections();
        let mut graph = Graph::with_capacity(params, 4);

        insert_node(&mut graph, 0, 1, 0)?;
        insert_node(&mut graph, 1, 1, 1)?;
        insert_node(&mut graph, 2, 1, 2)?;
        insert_node(&mut graph, 3, 1, 3)?;

        // Seed node 1 at capacity with node 2 (bidirectional)
        add_edge_if_missing(&mut graph, 1, 2, 1);
        add_edge_if_missing(&mut graph, 2, 1, 1);

        let new_node = NewNodeContext { id: 3, level: 1 };
        Ok(Self {
            graph,
            max_connections,
            new_node,
        })
    }

    /// Applies the given updates and returns the graph for assertions.
    fn apply_updates(
        mut self,
        updates: Vec<(StagedUpdate, Vec<usize>)>,
    ) -> Result<Graph, HnswError> {
        let mut applicator = CommitApplicator::new(&mut self.graph);
        let (reciprocated, _) =
            applicator.apply_neighbour_updates(updates, self.max_connections, self.new_node)?;
        applicator.apply_new_node_neighbours(
            self.new_node.id,
            self.new_node.level,
            reciprocated,
        )?;
        Ok(self.graph)
    }
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
    let ctx = EvictionTestContext::new(params_one_connection)?;
    let update = build_update(0, 1, vec![1], ctx.max_connections);
    let graph = ctx.apply_updates(vec![update])?;

    // Node 0 and node 1 should be linked
    assert_bidirectional_edge(&graph, 0, 1, 1);

    // Node 2's forward edge to node 1 should be scrubbed
    assert_no_edge(&graph, 2, 1, 1);

    // Node 1 should no longer link to node 2
    assert_no_edge(&graph, 1, 2, 1);

    Ok(())
}

mod deferred_scrub;
