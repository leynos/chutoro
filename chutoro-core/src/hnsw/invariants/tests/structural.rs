//! Structural regression tests for self-loops, duplicates, and entry points.

use super::*;
use rstest::rstest;

#[rstest]
#[case::single_node(1)]
#[case::two_nodes(2)]
#[case::four_nodes(4)]
fn no_self_loops_after_construction(#[case] node_count: usize) {
    let params = HnswParams::new(4, 8).expect("params");
    let mut graph = Graph::with_capacity(params, node_count);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert first");
    for id in 1..node_count {
        graph
            .attach_node(NodeContext {
                node: id,
                level: 0,
                sequence: id as u64,
            })
            .expect("attach node");
    }

    for origin in 0..node_count {
        for target in 0..node_count {
            if origin != target && target < node_count {
                graph
                    .node_mut(origin)
                    .expect("node")
                    .neighbours_mut(0)
                    .push(target);
            }
        }
    }

    assert!(helpers::has_no_self_loops(&graph));
}

#[test]
fn self_loop_is_detectable() {
    let params = HnswParams::new(4, 8).expect("params");
    let mut graph = Graph::with_capacity(params, 2);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert first");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach node");

    graph.node_mut(0).expect("node").neighbours_mut(0).push(0);

    assert!(
        !helpers::has_no_self_loops(&graph),
        "self-loop should be present for detection",
    );
}

#[rstest]
#[case::empty_neighbours(vec![])]
#[case::single_neighbour(vec![1])]
#[case::multiple_unique_neighbours(vec![1, 2, 3])]
fn neighbour_list_is_unique(#[case] neighbours: Vec<usize>) {
    let params = HnswParams::new(4, 8).expect("params");
    let mut graph = Graph::with_capacity(params, 5);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert first");
    for id in 1..5usize {
        graph
            .attach_node(NodeContext {
                node: id,
                level: 0,
                sequence: id as u64,
            })
            .expect("attach node");
    }

    for &neighbour in &neighbours {
        graph
            .node_mut(0)
            .expect("node")
            .neighbours_mut(0)
            .push(neighbour);
    }

    assert!(
        helpers::has_unique_neighbours(&graph),
        "neighbour list should have no duplicates",
    );
}

#[test]
fn duplicate_neighbour_is_detectable() {
    let params = HnswParams::new(4, 8).expect("params");
    let mut graph = Graph::with_capacity(params, 3);

    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("insert first");
    graph
        .attach_node(NodeContext {
            node: 1,
            level: 0,
            sequence: 1,
        })
        .expect("attach node");
    graph
        .attach_node(NodeContext {
            node: 2,
            level: 0,
            sequence: 2,
        })
        .expect("attach node");

    let neighbours = graph.node_mut(0).expect("node").neighbours_mut(0);
    neighbours.push(1);
    neighbours.push(1);

    assert!(
        !helpers::has_unique_neighbours(&graph),
        "duplicate should be present for detection",
    );
}

#[rstest]
#[case::single_node_level_0(vec![0], 0)]
#[case::single_node_level_1(vec![1], 1)]
#[case::two_nodes_same_level(vec![0, 0], 0)]
#[case::two_nodes_different_levels(vec![0, 2], 2)]
#[case::three_nodes_varying_levels(vec![1, 0, 2], 2)]
fn entry_point_has_max_level(#[case] levels: Vec<usize>, #[case] expected_entry_level: usize) {
    let max_level = *levels.iter().max().unwrap_or(&0);
    let params = HnswParams::new(4, 8)
        .expect("params")
        .with_max_level(max_level.saturating_add(1));
    let mut graph = Graph::with_capacity(params, levels.len());

    graph
        .insert_first(NodeContext {
            node: 0,
            level: levels[0],
            sequence: 0,
        })
        .expect("insert first");

    for (id, &level) in levels.iter().enumerate().skip(1) {
        graph
            .attach_node(NodeContext {
                node: id,
                level,
                sequence: id as u64,
            })
            .expect("attach node");
        graph.promote_entry(id, level);
    }

    let entry = graph.entry().expect("entry should exist");
    assert_eq!(
        entry.level, expected_entry_level,
        "entry point should have maximum level"
    );
    assert!(helpers::is_entry_point_valid(&graph));
}

#[test]
fn empty_graph_has_no_entry_point() {
    let params = HnswParams::new(4, 8).expect("params");
    let graph = Graph::with_capacity(params, 4);
    assert!(helpers::is_entry_point_valid(&graph));
}
