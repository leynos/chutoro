//! Test fixtures and helpers for trimming eviction scenarios.
//!
//! These helpers build graphs with specific neighbour configurations to
//! exercise the insertion executor's trimming and reciprocity restoration
//! logic.

use super::*;
use crate::hnsw::{
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext},
    params::{HnswParams, connection_limit_for_level},
    types::{InsertionPlan, LayerPlan, Neighbour},
};

pub(super) fn apply_insertion_with_trim(
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

pub(super) fn verify_post_trim_reciprocity(
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

pub(super) fn build_trimming_test_graph(
    params: &HnswParams,
    trimmed_neighbours: &[usize],
    reserve_id: usize,
    new_node_id: usize,
) -> Result<Graph, HnswError> {
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

    // new_node_id reserved for the insertion under test.
    let _ = new_node_id;

    Ok(graph)
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

pub(super) fn setup_reciprocal_edges_with_reserve(
    graph: &mut Graph,
    trimmed_neighbours: &[usize],
    evicted: usize,
    reserve_id: usize,
) {
    set_entry_neighbours(graph, trimmed_neighbours);

    for &neighbour in trimmed_neighbours {
        link_if_absent(graph, neighbour, 0);
    }

    link_if_absent(graph, evicted, reserve_id);
    link_if_absent(graph, reserve_id, evicted);
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
