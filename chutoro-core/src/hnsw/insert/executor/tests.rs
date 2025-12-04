use super::*;
use crate::hnsw::insert::{reconciliation::EdgeReconciler, test_helpers::TestHelpers, types};
use crate::hnsw::{
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext},
    params::{HnswParams, connection_limit_for_level},
    types::{InsertionPlan, LayerPlan, Neighbour},
};
use rstest::rstest;

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

#[rstest]
#[case::evicts_tail(vec![1, 3], 1)]
#[case::evicts_tail_wider(vec![1, 3, 4, 5], 2)]
fn trimming_eviction_restores_reciprocity(
    #[case] trimmed_neighbours: Vec<usize>,
    #[case] max_connections: usize,
) -> Result<(), HnswError> {
    assert!(
        !trimmed_neighbours.is_empty(),
        "trimmed_neighbours must be non-empty to exercise eviction fallback",
    );

    let params = HnswParams::new(max_connections, max_connections * 4)?;
    let TrimScenario {
        mut graph,
        evicted,
        new_node_id,
    } = build_trim_eviction_scenario(&params, &trimmed_neighbours)?;

    apply_insertion_with_trim(&mut graph, &params, new_node_id, trimmed_neighbours.clone())?;

    assert_reciprocity_restored(&graph, &params, new_node_id, evicted);

    Ok(())
}

fn apply_insertion_with_trim(
    graph: &mut Graph,
    params: &HnswParams,
    new_node_id: usize,
    trimmed_neighbours: Vec<usize>,
) -> Result<(), HnswError> {
    let reserve_id = new_node_id.saturating_add(1);
    let mut executor = InsertionExecutor::new(graph);
    let plan = InsertionPlan {
        layers: vec![LayerPlan {
            level: 0,
            neighbours: vec![Neighbour {
                id: 0,
                distance: 0.0,
            }],
        }],
    };

    let (prepared, trim_jobs) = executor.apply(
        NodeContext {
            node: new_node_id,
            level: 0,
            sequence: (reserve_id as u64) + 1,
        },
        ApplyContext { params, plan },
    )?;

    assert_eq!(
        trim_jobs.len(),
        1,
        "only the entry node should require trim"
    );
    let job = trim_jobs.into_iter().next().expect("trim job expected");
    assert_eq!(job.node, 0, "trim must target the entry node");
    let trim_result = TrimResult {
        node: job.node,
        ctx: job.ctx,
        neighbours: trimmed_neighbours,
    };

    executor.commit(prepared, vec![trim_result])
}

fn assert_reciprocity_restored(
    graph: &Graph,
    params: &HnswParams,
    new_node_id: usize,
    evicted: usize,
) {
    let connection_limit = connection_limit_for_level(0, params.max_connections());
    let entry = graph.node(0).expect("entry node available");
    let entry_neighbours = entry.neighbours(0);
    assert!(
        entry_neighbours.contains(&new_node_id),
        "reciprocity pass should reintroduce the new node even after trim eviction",
    );
    assert!(
        !entry_neighbours.contains(&evicted),
        "evicted neighbour should be removed to honour capacity constraints",
    );
    assert!(
        entry_neighbours.len() <= connection_limit,
        "entry degree should respect the base-layer limit after reconciliation",
    );

    let new_node = graph
        .node(new_node_id)
        .expect("new node must be attached after commit");
    assert!(
        new_node.neighbours(0).contains(&0),
        "new node should have a reciprocal edge to the entry node",
    );

    if let Some(evicted_node) = graph.node(evicted) {
        assert!(
            !evicted_node.neighbours(0).contains(&0),
            "forward edge from evicted neighbour should be removed",
        );
    }
}

#[derive(Debug)]
struct TrimScenario {
    graph: Graph,
    evicted: usize,
    new_node_id: usize,
}

fn build_trim_eviction_scenario(
    params: &HnswParams,
    trimmed_neighbours: &[usize],
) -> Result<TrimScenario, HnswError> {
    let new_node_id = trimmed_neighbours
        .iter()
        .copied()
        .max()
        .expect("trimmed_neighbours asserted as non-empty")
        .saturating_add(1);
    let reserve_id = new_node_id.saturating_add(1);
    let capacity = reserve_id.saturating_add(1);
    let mut graph = Graph::with_capacity(params.clone(), capacity);
    graph.insert_first(NodeContext {
        node: 0,
        level: 0,
        sequence: 0,
    })?;

    attach_neighbours(&mut graph, trimmed_neighbours, 1)?;
    graph.attach_node(NodeContext {
        node: reserve_id,
        level: 0,
        sequence: (trimmed_neighbours.len() + 1) as u64,
    })?;

    set_entry_neighbours(&mut graph, trimmed_neighbours);

    let evicted = *trimmed_neighbours
        .last()
        .expect("trimmed_neighbours asserted as non-empty");

    for &neighbour in trimmed_neighbours {
        link_if_absent(&mut graph, neighbour, 0);
    }

    link_if_absent(&mut graph, evicted, reserve_id);
    link_if_absent(&mut graph, reserve_id, evicted);

    Ok(TrimScenario {
        graph,
        evicted,
        new_node_id,
    })
}

fn attach_neighbours(
    graph: &mut Graph,
    neighbours: &[usize],
    start_sequence: u64,
) -> Result<(), HnswError> {
    for (offset, &id) in neighbours.iter().enumerate() {
        graph.attach_node(NodeContext {
            node: id,
            level: 0,
            sequence: start_sequence + offset as u64,
        })?;
    }
    Ok(())
}

fn set_entry_neighbours(graph: &mut Graph, neighbours: &[usize]) {
    let entry_neighbours = graph.node_mut(0).expect("entry present").neighbours_mut(0);
    entry_neighbours.clear();
    entry_neighbours.extend(neighbours.iter().copied());
}

fn link_if_absent(graph: &mut Graph, origin: usize, target: usize) {
    let list = graph
        .node_mut(origin)
        .unwrap_or_else(|| panic!("node {origin} should be present"))
        .neighbours_mut(0);
    if !list.contains(&target) {
        list.push(target);
    }
}
