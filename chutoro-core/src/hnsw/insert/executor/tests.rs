use super::*;
use crate::hnsw::insert::{reconciliation::EdgeReconciler, test_helpers::TestHelpers, types};
use crate::hnsw::{
    graph::{Graph, NodeContext},
    params::HnswParams,
};

#[test]
fn ensure_reverse_edge_evicts_and_scrubs_forward_link() {
    let params = HnswParams::new(1, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 3);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 1,
            sequence: 0,
        })
        .expect("insert entry");
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

    // Forward edges: 0 -> 1, 2 -> 1; target (1) is at capacity and prefers 2.
    graph.node_mut(0).unwrap().neighbours_mut(1).push(1);
    graph.node_mut(1).unwrap().neighbours_mut(1).push(2);
    graph.node_mut(2).unwrap().neighbours_mut(1).push(1);

    let mut reconciler = EdgeReconciler::new(&mut graph);
    let ensured = reconciler.ensure_reverse_edge(
        &types::UpdateContext {
            origin: 0,
            level: 1,
            max_connections: 1,
        },
        1,
    );

    assert!(ensured, "reverse edge should be ensured even when evicting");

    let target = reconciler.graph.node(1).unwrap();
    assert_eq!(target.neighbours(1), &[0]);

    let evicted = reconciler.graph.node(2).unwrap();
    assert!(
        !evicted.neighbours(1).contains(&1),
        "evicted neighbour should lose its forward edge to maintain reciprocity",
    );

    let origin = reconciler.graph.node(0).unwrap();
    assert!(origin.neighbours(1).contains(&1));
}

#[test]
fn ensure_new_node_reciprocity_removes_one_way_edges() {
    let params = HnswParams::new(1, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 2);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert entry");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach node 1");

    graph.node_mut(1).unwrap().neighbours_mut(0).push(0);

    let mut enforcer = ReciprocityEnforcer::new(&mut graph);
    enforcer.ensure_reciprocity_for_touched(&[(1, 0)], 1);

    let node0 = enforcer.graph.node(0).unwrap();
    let node1 = enforcer.graph.node(1).unwrap();

    assert!(node0.neighbours(0).contains(&1));
    assert!(node1.neighbours(0).contains(&0));
}

#[test]
fn ensure_reciprocity_for_touched_heals_existing_one_way() {
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 3);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert entry");
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

    // One-way edge from node 2 to node 0.
    graph.node_mut(2).unwrap().neighbours_mut(0).push(0);

    let mut enforcer = ReciprocityEnforcer::new(&mut graph);
    enforcer.ensure_reciprocity_for_touched(&[(2, 0)], 2);

    let node0 = enforcer.graph.node(0).unwrap();
    let node2 = enforcer.graph.node(2).unwrap();

    assert!(node0.neighbours(0).contains(&2));
    assert!(node2.neighbours(0).contains(&0));
}

#[test]
fn enforce_bidirectional_all_adds_upper_layer_backlink() {
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 2);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 1,
            sequence: 0,
        })
        .expect("insert entry");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 1,
            sequence: 1,
        })
        .expect("attach node 1");

    graph.node_mut(0).unwrap().neighbours_mut(1).push(1);

    TestHelpers::new(&mut graph).enforce_bidirectional_all(2);

    let node0 = graph.node(0).unwrap();
    let node1 = graph.node(1).unwrap();

    assert!(node0.neighbours(1).contains(&1));
    assert!(node1.neighbours(1).contains(&0));
}

#[test]
fn enforce_bidirectional_all_removes_invalid_upper_edge() {
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 2);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 1,
            sequence: 0,
        })
        .expect("insert entry");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach node 1");

    // One-way edge exists at level 1, but target only has level 0.
    graph.node_mut(0).unwrap().neighbours_mut(1).push(1);

    TestHelpers::new(&mut graph).enforce_bidirectional_all(2);

    let node0 = graph.node(0).unwrap();
    assert!(node0.neighbours(1).is_empty());
}
