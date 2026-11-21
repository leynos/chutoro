//! Tests reverse-edge removal during insertion reconciliation.

use super::{EdgeReconciler, UpdateContext};
use crate::hnsw::{
    error::HnswError,
    graph::{Graph, NodeContext},
    params::HnswParams,
};

fn graph_with_two_nodes(level: usize) -> Result<Graph, HnswError> {
    let params = HnswParams::new(2, 4)?;
    let mut graph = Graph::with_capacity(params, 2);
    graph.insert_first(NodeContext {
        node: 0,
        level,
        sequence: 0,
    })?;
    graph.attach_node(NodeContext {
        node: 1,
        level,
        sequence: 1,
    })?;
    Ok(graph)
}

fn update_context(level: usize) -> UpdateContext {
    UpdateContext {
        origin: 0,
        level,
        max_connections: 2,
    }
}

#[test]
fn remove_reverse_edge_reports_a_base_layer_target_it_isolates() {
    let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
    graph
        .node_mut(1)
        .expect("target node must be present")
        .neighbours_mut(0)
        .expect("target must have a base layer")
        .push(0);

    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(reconciler.remove_reverse_edge(&update_context(0), 1));
    assert_eq!(
        reconciler
            .graph
            .node(1)
            .expect("target node must be present")
            .neighbours(0),
        [] as [usize; 0]
    );
}

#[test]
fn remove_reverse_edge_does_not_report_an_upper_layer_target_as_isolated() {
    let mut graph = graph_with_two_nodes(1).expect("test graph must be valid");
    graph
        .node_mut(1)
        .expect("target node must be present")
        .neighbours_mut(1)
        .expect("target must have an upper layer")
        .push(0);

    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(1), 1));
    assert_eq!(
        reconciler
            .graph
            .node(1)
            .expect("target node must be present")
            .neighbours(1),
        [] as [usize; 0]
    );
}

#[test]
fn remove_reverse_edge_returns_false_when_the_reverse_edge_is_missing() {
    let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(0), 1));
}

#[test]
fn remove_reverse_edge_returns_false_when_target_or_level_is_missing() {
    let mut graph = graph_with_two_nodes(0).expect("test graph must be valid");
    let mut reconciler = EdgeReconciler::new(&mut graph);

    assert!(!reconciler.remove_reverse_edge(&update_context(0), 2));
    assert!(!reconciler.remove_reverse_edge(&update_context(1), 1));
}