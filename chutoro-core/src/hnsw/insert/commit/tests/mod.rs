//! Commit-path tests for neighbour updates and deferred scrubs.

use super::super::limits;
use super::super::test_helpers::{add_edge_if_missing, assert_bidirectional_edge, assert_no_edge};
use super::CommitApplicator;
use crate::hnsw::{
    error::HnswError,
    graph::{EdgeContext, Graph, NodeContext},
    insert::types::{NewNodeContext, StagedUpdate},
    params::HnswParams,
};
use rstest::{fixture, rstest};

/// Parameters allowing two connections per node; fallible so tests, rather
/// than the fixture, surface any configuration error.
#[fixture]
fn params_two_connections() -> Result<HnswParams, HnswError> {
    HnswParams::new(2, 4)
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
    #[from(params_two_connections)] params_res: Result<HnswParams, HnswError>,
) {
    let params = params_res.expect("params should be valid for tests");
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 3);

    insert_node(&mut graph, 0, level, 0).expect("insert node 0");
    insert_node(&mut graph, 1, level, 1).expect("insert node 1");
    insert_node(&mut graph, 2, level, 2).expect("insert node 2");

    add_edge_if_missing(&mut graph, 0, 1, level);
    add_edge_if_missing(&mut graph, 1, 0, level);

    let update = build_update(0, level, vec![1, 2], max_connections);
    let new_node = NewNodeContext { id: 2, level };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) = applicator
        .apply_neighbour_updates(vec![update], max_connections, new_node)
        .expect("apply neighbour updates");
    applicator
        .apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)
        .expect("apply new-node neighbours");

    assert_bidirectional_edge!(&graph, 0, 2, level);
    assert_bidirectional_edge!(&graph, 0, 1, level);
}

#[test]
fn commit_updates_scrub_evicted_forward_edge() {
    let params = HnswParams::new(1, 4).expect("test parameters must be valid");
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 4);

    insert_node(&mut graph, 0, 1, 0).expect("insert node 0");
    insert_node(&mut graph, 1, 1, 1).expect("insert node 1");
    insert_node(&mut graph, 2, 1, 2).expect("insert node 2");
    insert_node(&mut graph, 3, 1, 3).expect("insert node 3");

    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    let update = build_update(0, 1, vec![1], max_connections);
    let new_node = NewNodeContext { id: 3, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) = applicator
        .apply_neighbour_updates(vec![update], max_connections, new_node)
        .expect("apply neighbour updates");
    applicator
        .apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)
        .expect("apply new-node neighbours");

    let limit = limits::compute_connection_limit(1, max_connections);
    for node_id in [0, 1, 2, 3] {
        let node_msg = format!("node {node_id} should exist");
        let node = graph.node(node_id).expect(&node_msg);
        assert!(
            node.neighbours(1).len() <= limit,
            "expected node {node_id} to respect level-1 connection limit",
        );
    }

    assert_bidirectional_edge!(&graph, 0, 1, 1);
    assert_no_edge(&graph, 2, 1, 1);
    assert_no_edge(&graph, 1, 2, 1);
}

#[rstest]
fn commit_updates_report_missing_origin(
    #[from(params_two_connections)] params_res: Result<HnswParams, HnswError>,
) {
    let params = params_res.expect("params should be valid for tests");
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

/// Parameters allowing a single connection per node; fallible so tests, rather
/// than the fixture, surface any configuration error.
#[fixture]
fn params_one_connection() -> Result<HnswParams, HnswError> {
    HnswParams::new(1, 4)
}

/// Asserts that every edge in the graph has its reverse edge at every level.
fn assert_graph_bidirectional(graph: &Graph, node_count: usize) {
    for node_id in 0..node_count {
        let Some(node) = graph.node(node_id) else {
            panic!("node {node_id} should exist");
        };
        for level in 0..node.level_count() {
            assert_level_edges_reciprocated(graph, node_id, node.neighbours(level), level);
        }
    }
}

/// Asserts that each listed neighbour links back to `node_id` at `level`.
fn assert_level_edges_reciprocated(
    graph: &Graph,
    node_id: usize,
    neighbours: &[usize],
    level: usize,
) {
    for &neighbour in neighbours {
        let Some(other) = graph.node(neighbour) else {
            panic!("neighbour {neighbour} should exist");
        };
        assert!(
            level < other.level_count() && other.neighbours(level).contains(&node_id),
            "edge {node_id}->{neighbour} at level {level} has no reverse edge",
        );
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
    #[from(params_one_connection)] params_res: Result<HnswParams, HnswError>,
) {
    let params = params_res.expect("params should be valid for tests");
    let ctx = EvictionTestContext::new(params).expect("eviction fixture must initialize");
    let update = build_update(0, 1, vec![1], ctx.max_connections);
    let graph = ctx
        .apply_updates(vec![update])
        .expect("apply eviction update");

    // Node 0 and node 1 should be linked
    assert_bidirectional_edge!(&graph, 0, 1, 1);

    // Node 2's forward edge to node 1 should be scrubbed
    assert_no_edge(&graph, 2, 1, 1);

    // Node 1 should no longer link to node 2
    assert_no_edge(&graph, 1, 2, 1);
}

mod deferred_scrub;

/// Regression test: replacing a neighbour whose only base-layer edge was to
/// the origin must not leave a dangling reverse edge.
///
/// Removing node 1 from node 0's list isolates node 1 at the base layer, so
/// the connectivity healer links it back to the entry node, which is node 0
/// itself. Removed-edge reconciliation therefore has to run after node 0's
/// neighbour list is written back; healing against the stale pre-write-back
/// list is clobbered by the write-back, leaving `1 -> 0` without `0 -> 1`.
#[rstest]
fn isolation_replacement_keeps_bidirectionality(
    params_two_connections: HnswParams,
) -> Result<(), HnswError> {
    let max_connections = params_two_connections.max_connections();
    let mut graph = Graph::with_capacity(params_two_connections, 3);

    insert_node(&mut graph, 0, 0, 0)?;
    insert_node(&mut graph, 1, 0, 1)?;
    insert_node(&mut graph, 2, 0, 2)?;

    add_edge_if_missing(&mut graph, 0, 1, 0);
    add_edge_if_missing(&mut graph, 1, 0, 0);

    // Node 0 replaces neighbour 1 with neighbour 2, isolating node 1.
    let update = build_update(0, 0, vec![2], max_connections);
    let new_node = NewNodeContext { id: 2, level: 0 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    for node_id in 0..3usize {
        let node = graph.node(node_id).expect("node exists");
        for &neighbour in node.neighbours(0) {
            let other = graph.node(neighbour).expect("neighbour exists");
            assert!(
                other.neighbours(0).contains(&node_id),
                "edge {node_id}->{neighbour} has no reverse edge",
            );
        }
    }
    Ok(())
}
