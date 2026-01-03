//! Commit-path tests for neighbour updates and deferred scrubs.

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

fn add_edge(graph: &mut Graph, from: usize, to: usize, level: usize) {
    let list = graph
        .node_mut(from)
        .unwrap_or_else(|| panic!("node {from} should exist"))
        .neighbours_mut(level);
    if !list.contains(&to) {
        list.push(to);
    }
}

fn assert_bidirectional_edge(graph: &Graph, node_a: usize, node_b: usize, level: usize) {
    let a = graph
        .node(node_a)
        .unwrap_or_else(|| panic!("node {node_a} should exist"));
    let b = graph
        .node(node_b)
        .unwrap_or_else(|| panic!("node {node_b} should exist"));
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

fn assert_no_edge(graph: &Graph, from: usize, to: usize, level: usize) {
    if let Some(node) = graph.node(from) {
        if level < node.level_count() {
            assert!(
                !node.neighbours(level).contains(&to),
                "unexpected edge {from}->{to} at level {level}",
            );
        }
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
    params_two_connections: HnswParams,
) -> Result<(), HnswError> {
    let max_connections = params_two_connections.max_connections();
    let mut graph = Graph::with_capacity(params_two_connections.clone(), 3);

    insert_node(&mut graph, 0, level, 0)?;
    insert_node(&mut graph, 1, level, 1)?;
    insert_node(&mut graph, 2, level, 2)?;

    add_edge(&mut graph, 0, 1, level);
    add_edge(&mut graph, 1, 0, level);

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

    add_edge(&mut graph, 1, 2, 1);
    add_edge(&mut graph, 2, 1, 1);

    let update = build_update(0, 1, vec![1], max_connections);
    let new_node = NewNodeContext { id: 3, level: 1 };

    let mut applicator = CommitApplicator::new(&mut graph);
    let (reciprocated, _) =
        applicator.apply_neighbour_updates(vec![update], max_connections, new_node)?;
    applicator.apply_new_node_neighbours(new_node.id, new_node.level, reciprocated)?;

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
