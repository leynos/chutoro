//! Tests for graph deletion and reconnection helpers.

use rstest::{fixture, rstest};

use crate::hnsw::{HnswError, graph::NodeContext, graph::core::Graph, params::HnswParams};

/// Creates default parameters for most deletion tests (max_connections=2, ef_construction=4).
#[fixture]
fn basic_params() -> HnswParams {
    HnswParams::new(2, 4).expect("params must be valid")
}

/// Creates a graph with capacity 3 using basic params.
#[fixture]
fn small_graph(basic_params: HnswParams) -> Graph {
    Graph::with_capacity(basic_params, 3)
}

/// Creates restricted parameters for disconnection tests (max_connections=1, ef_construction=1).
#[fixture]
fn restricted_params() -> HnswParams {
    HnswParams::new(1, 1).expect("params must be valid")
}

#[rstest]
fn delete_node_reconnects_neighbours_and_preserves_reachability(mut small_graph: Graph) {
    small_graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert entry");
    small_graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach first neighbour");
    small_graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .expect("attach second neighbour");
    small_graph.try_add_bidirectional_edge(0, 1, 0);
    small_graph.try_add_bidirectional_edge(1, 2, 0);

    let deleted = small_graph.delete_node(1).expect("delete must succeed");

    assert!(deleted, "node should be removed");
    assert!(
        small_graph.node(1).is_none(),
        "slot 1 must be cleared after deletion"
    );
    let node0 = small_graph.node(0).expect("node 0 must remain");
    assert_eq!(node0.neighbours(0), &[2], "node 0 must connect to node 2");
    let node2 = small_graph.node(2).expect("node 2 must remain");
    assert_eq!(node2.neighbours(0), &[0], "node 2 must connect to node 0");
    assert_eq!(small_graph.entry().map(|entry| entry.node), Some(0));
}

#[rstest]
fn delete_node_returns_ok_false_for_missing_node(mut small_graph: Graph) {
    small_graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert entry");

    let first_delete = small_graph.delete_node(0).expect("delete existing node");
    assert!(
        first_delete,
        "expected Ok(true) when deleting an existing node"
    );

    let second_delete = small_graph.delete_node(0);
    assert!(
        matches!(second_delete, Ok(false)),
        "expected Ok(false) when deleting a missing node, got {second_delete:?}",
    );

    let third_delete = small_graph.delete_node(0);
    assert!(
        matches!(third_delete, Ok(false)),
        "expected Ok(false) on repeated deletes of a missing node, got {third_delete:?}",
    );
}

#[rstest]
fn delete_node_returns_invalid_parameters_for_out_of_bounds_index(mut small_graph: Graph) {
    let result = small_graph.delete_node(5);

    match result {
        Err(HnswError::InvalidParameters { .. }) => {}
        other => panic!("expected Err(HnswError::InvalidParameters {{ .. }}), got {other:?}",),
    }
}

#[rstest]
fn delete_node_reverts_when_it_would_disconnect_graph(restricted_params: HnswParams) {
    let mut graph = Graph::with_capacity(restricted_params, 5);
    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert entry");
    for (node, sequence) in [(1_usize, 1_u64), (2, 2), (3, 3), (4, 4)] {
        graph
            .attach_node(NodeContext {
                node,
                level: 0,
                sequence,
            })
            .expect("attach node");
    }
    graph.try_add_bidirectional_edge(0, 1, 0);
    graph.try_add_bidirectional_edge(0, 2, 0);
    graph.try_add_bidirectional_edge(0, 3, 0);
    graph.try_add_bidirectional_edge(1, 4, 0);
    graph.try_add_bidirectional_edge(2, 4, 0);

    let result = graph.delete_node(0);

    let err = result.expect_err("delete must fail to preserve reachability");
    match err {
        HnswError::GraphInvariantViolation { .. } => {}
        other => panic!("expected GraphInvariantViolation, got {other:?}"),
    }
    assert!(
        graph.node(0).is_some(),
        "failed deletion must restore the removed node"
    );
    let node1 = graph.node(1).expect("node 1 must remain");
    assert!(
        node1.neighbours(0).contains(&0),
        "node 1 should retain its link to the entry"
    );
    assert_eq!(
        graph.entry().map(|entry| entry.node),
        Some(0),
        "entry point must roll back on failure"
    );
}
