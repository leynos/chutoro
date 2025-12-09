//! Tests the insertion executor's edge reconciliation: reverse-edge eviction,
//! inline reciprocity after trimming, fallback linking, and bidirectional
//! healing. Scenarios build tiny graphs manually with deterministic sequences
//! to validate post-commit graph state and degree limits without concurrency.

mod trimming_fixtures;

use super::*;
use crate::hnsw::insert::{reconciliation::EdgeReconciler, test_helpers::TestHelpers, types};
use crate::hnsw::{
    error::HnswError,
    graph::{ApplyContext, Graph, NodeContext},
    params::HnswParams,
    types::{InsertionPlan, LayerPlan, Neighbour},
};
use rstest::rstest;
use trimming_fixtures::{
    apply_insertion_with_trim, build_trimming_test_graph, setup_reciprocal_edges_with_reserve,
    verify_post_trim_reciprocity,
};

fn setup_basic_graph(max_connections: usize, ef_construction: usize, capacity: usize) -> Graph {
    let params =
        HnswParams::new(max_connections, ef_construction).expect("params should be valid in tests");
    Graph::with_capacity(params, capacity)
}

fn insert_entry_node(graph: &mut Graph, level: usize) {
    graph
        .insert_first(NodeContext {
            node: 0,
            level,
            sequence: 0,
        })
        .expect("insert entry");
}

fn attach_test_node(graph: &mut Graph, node: usize, level: usize, sequence: u64) {
    graph
        .attach_node(NodeContext {
            node,
            level,
            sequence,
        })
        .expect("attach node");
}

fn add_edge(graph: &mut Graph, from: usize, to: usize, level: usize) {
    let list = graph
        .node_mut(from)
        .unwrap_or_else(|| panic!("node {from} should be present"))
        .neighbours_mut(level);
    if !list.contains(&to) {
        list.push(to);
    }
}

fn assert_bidirectional_edge(graph: &Graph, node_a: usize, node_b: usize, level: usize) {
    let a = graph
        .node(node_a)
        .unwrap_or_else(|| panic!("node {node_a} should be present"));
    let b = graph
        .node(node_b)
        .unwrap_or_else(|| panic!("node {node_b} should be present"));
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

    // Apply deferred scrubs to remove the evicted node's forward edge.
    reconciler.apply_deferred_scrubs(1);

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

#[rstest]
#[case::repairs_base_layer_one_way_edge(2, 0, None, 0)]
#[case::removes_invalid_upper_layer_edge(1, 1, Some(vec![1]), 1)]
fn commit_inlines_reciprocity(
    #[case] max_connections: usize,
    #[case] seed_edge_level: usize,
    #[case] trim_override: Option<Vec<usize>>,
    #[case] new_node_level: usize,
) -> Result<(), HnswError> {
    let params = HnswParams::new(max_connections, 4)?;
    let entry_level = new_node_level.max(seed_edge_level);
    let mut graph = setup_basic_graph(max_connections, 4, 4);
    insert_entry_node(&mut graph, entry_level);

    attach_test_node(&mut graph, 1, 0, 1);

    add_edge(&mut graph, 0, 1, seed_edge_level);

    let mut layers = vec![LayerPlan {
        level: 0,
        neighbours: vec![Neighbour {
            id: 0,
            distance: 0.0,
        }],
    }];
    if new_node_level > 0 {
        layers.push(LayerPlan {
            level: new_node_level,
            neighbours: vec![Neighbour {
                id: 0,
                distance: 0.0,
            }],
        });
    }

    let mut executor = InsertionExecutor::new(&mut graph);
    let (prepared, trim_jobs) = executor.apply(
        NodeContext {
            node: 2,
            level: new_node_level,
            sequence: 2,
        },
        ApplyContext {
            params: &params,
            plan: InsertionPlan { layers },
        },
    )?;

    let trims: Vec<TrimResult> = trim_jobs
        .iter()
        .map(|job| TrimResult {
            node: job.node,
            ctx: job.ctx,
            neighbours: trim_override
                .clone()
                .unwrap_or_else(|| job.candidates.clone()),
        })
        .collect();

    executor.commit(prepared, trims)?;

    assert_bidirectional_edge(&graph, 0, 2, 0);
    if new_node_level > 0 {
        assert_bidirectional_edge(&graph, 0, 2, new_node_level);
        assert_no_edge(&graph, 0, 1, seed_edge_level);
        assert_no_edge(&graph, 1, 0, seed_edge_level);
    } else {
        assert_bidirectional_edge(&graph, 0, 1, 0);
    }

    Ok(())
}

#[test]
fn enforce_bidirectional_all_adds_upper_layer_backlink() {
    let mut graph = setup_basic_graph(2, 4, 2);
    insert_entry_node(&mut graph, 1);
    attach_test_node(&mut graph, 1, 1, 1);

    add_edge(&mut graph, 0, 1, 1);

    TestHelpers::new(&mut graph).enforce_bidirectional_all(2);

    assert_bidirectional_edge(&graph, 0, 1, 1);
}

#[test]
fn enforce_bidirectional_all_removes_invalid_upper_edge() {
    let mut graph = setup_basic_graph(2, 4, 2);
    insert_entry_node(&mut graph, 1);
    attach_test_node(&mut graph, 1, 0, 1);

    // One-way edge exists at level 1, but target only has level 0.
    add_edge(&mut graph, 0, 1, 1);

    TestHelpers::new(&mut graph).enforce_bidirectional_all(2);

    assert_no_edge(&graph, 0, 1, 1);
    assert_no_edge(&graph, 1, 0, 1);
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
    let new_node_id = trimmed_neighbours
        .iter()
        .copied()
        .max()
        .expect("trimmed_neighbours asserted as non-empty")
        .saturating_add(1);
    let reserve_id = new_node_id.saturating_add(1);
    let evicted = *trimmed_neighbours
        .last()
        .expect("trimmed_neighbours asserted as non-empty");

    let mut graph =
        build_trimming_test_graph(&params, &trimmed_neighbours, reserve_id, new_node_id)?;
    setup_reciprocal_edges_with_reserve(&mut graph, &trimmed_neighbours, evicted, reserve_id);

    apply_insertion_with_trim(&mut graph, &params, new_node_id, trimmed_neighbours.clone())?;

    verify_post_trim_reciprocity(&graph, &params, new_node_id, evicted);

    Ok(())
}
