//! Eviction-path tests for deferred scrubs and ordering guarantees.

use super::*;
use rstest::{fixture, rstest};

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

#[rstest]
fn eviction_scrubs_orphaned_forward_edge(
    params_one_connection: HnswParams,
) -> Result<(), HnswError> {
    let ctx = EvictionTestContext::new(params_one_connection)?;
    let update = build_update(0, 1, vec![1], ctx.max_connections);
    let graph = ctx.apply_updates(vec![update])?;

    assert_bidirectional_edge(&graph, 0, 1, 1);
    assert_no_edge(&graph, 2, 1, 1);
    assert_no_edge(&graph, 1, 2, 1);

    Ok(())
}

#[rstest]
fn benign_deferred_scrub_is_noop_when_edge_already_removed(
    params_one_connection: HnswParams,
) -> Result<(), HnswError> {
    let max_connections = params_one_connection.max_connections();
    let mut graph = Graph::with_capacity(params_one_connection, 5);

    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;
    insert_node(&mut graph, 4, 1, 4)?;

    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);

    let update1 = build_update(0, 1, vec![1], max_connections);
    let update2 = build_update(2, 1, vec![3], max_connections);
    let new_node = NewNodeContext { id: 4, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update1, update2], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    assert_bidirectional_edge(&graph, 0, 1, 1);
    assert_no_edge(&graph, 2, 1, 1);
    assert_no_edge(&graph, 1, 2, 1);
    assert_bidirectional_edge(&graph, 2, 3, 1);

    Ok(())
}

#[rstest]
fn eviction_skips_scrub_if_reciprocity_restored(
    params_one_connection: HnswParams,
) -> Result<(), HnswError> {
    let ctx = EvictionTestContext::new(params_one_connection)?;
    let update1 = build_update(0, 1, vec![1], ctx.max_connections);
    let update2 = build_update(2, 1, vec![1], ctx.max_connections);
    let graph = ctx.apply_updates(vec![update1, update2])?;

    assert_has_edge(&graph, 2, 1, 1);
    Ok(())
}

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

    add_edge_if_missing(&mut graph, 1, 2, 1);
    add_edge_if_missing(&mut graph, 2, 1, 1);
    add_edge_if_missing(&mut graph, 3, 4, 1);
    add_edge_if_missing(&mut graph, 4, 3, 1);

    let update1 = build_update(0, 1, vec![1], max_connections);
    let update2 = build_update(5, 1, vec![3], max_connections);
    let new_node = NewNodeContext { id: 6, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update1, update2], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    assert_no_edge(&graph, 2, 1, 1);
    assert_no_edge(&graph, 4, 3, 1);
    assert_bidirectional_edge(&graph, 0, 1, 1);
    assert_bidirectional_edge(&graph, 5, 3, 1);

    Ok(())
}

#[rstest]
fn eviction_respects_furthest_first_ordering() -> Result<(), HnswError> {
    let params = HnswParams::new(2, 4)?;
    let max_connections = params.max_connections();
    let mut graph = Graph::with_capacity(params, 5);

    insert_node(&mut graph, 0, 1, 0)?;
    insert_node(&mut graph, 1, 1, 1)?;
    insert_node(&mut graph, 2, 1, 2)?;
    insert_node(&mut graph, 3, 1, 3)?;
    insert_node(&mut graph, 4, 1, 4)?;

    let node1 = graph.node_mut(1).expect("node 1 should exist");
    node1.neighbours_mut(1).push(2);
    node1.neighbours_mut(1).push(3);

    add_edge_if_missing(&mut graph, 2, 1, 1);
    add_edge_if_missing(&mut graph, 3, 1, 1);

    let update = build_update(0, 1, vec![1], max_connections);
    let new_node = NewNodeContext { id: 4, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

    assert_no_edge(&graph, 1, 2, 1);
    assert_no_edge(&graph, 2, 1, 1);
    assert_has_edge(&graph, 1, 3, 1);
    assert_bidirectional_edge(&graph, 0, 1, 1);

    Ok(())
}
