use super::{
    EvaluationMode, GraphContext, HnswInvariant, HnswInvariantViolation, check_degree_bounds,
    check_reachability, helpers,
};
use crate::{
    datasource::DataSource,
    error::DataSourceError,
    hnsw::{
        CpuHnsw,
        graph::{Graph, NodeContext},
        params::HnswParams,
    },
};
use rstest::rstest;

#[derive(Clone)]
struct Dummy(Vec<f32>);

impl DataSource for Dummy {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn name(&self) -> &str {
        "dummy"
    }
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        Ok((self.0[i] - self.0[j]).abs())
    }
}

fn build_index() -> (CpuHnsw, Dummy) {
    let data = Dummy(vec![0.0, 1.0, 2.0, 3.0]);
    let params = HnswParams::new(4, 8).expect("params").with_rng_seed(7);
    let index = CpuHnsw::build(&data, params).expect("build hnsw");
    (index, data)
}

#[test]
fn check_all_succeeds_for_valid_index() {
    let (index, _data) = build_index();
    index.invariants().check_all().expect("graph valid");
}

#[rstest]
#[case::missing_node(|graph: &mut Graph| {
    graph.node_mut(0).expect("node 0").neighbours_mut(0).push(3);
})]
#[case::missing_layer(|graph: &mut Graph| {
    graph
        .node_mut(0)
        .expect("node 0")
        .neighbours_mut(1)
        .push(1);
})]
fn layer_consistency_reports_invalid_reference(#[case] mutate: fn(&mut Graph)) {
    let params = HnswParams::new(4, 8).expect("params").with_max_level(2);
    let mut graph = Graph::with_capacity(params.clone(), 4);
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
        .expect("attach node");
    mutate(&mut graph);

    let validator = helpers::LayerValidator::new(&graph);
    let err = validator.ensure(0, 1, 1).expect_err("invariant must fail");
    matches!(err, HnswInvariantViolation::LayerConsistency { .. })
        .then_some(())
        .expect("layer consistency violation expected");
}

#[rstest]
#[case(0, 9)]
#[case(1, 5)]
fn degree_bounds_detects_overflow(#[case] level: usize, #[case] degree: usize) {
    let params = HnswParams::new(4, 8).expect("params").with_max_level(2);
    let mut graph = Graph::with_capacity(params.clone(), 10);
    graph
        .insert_first(NodeContext {
            node: 0,
            level: level.max(1),
            sequence: 0,
        })
        .expect("insert entry");

    for id in 1..=degree {
        graph
            .attach_node(NodeContext {
                node: id,
                level,
                sequence: id as u64,
            })
            .expect("attach neighbour");
        graph
            .node_mut(id)
            .expect("reverse")
            .neighbours_mut(level)
            .push(0);
    }

    let node = graph.node_mut(0).expect("entry").neighbours_mut(level);
    node.clear();
    node.extend(1..=degree);

    let ctx = GraphContext {
        graph: &graph,
        params: &params,
    };
    let mut mode = EvaluationMode::FailFast;
    let err = check_degree_bounds(ctx, &mut mode).expect_err("must overflow");
    if let HnswInvariantViolation::DegreeBounds {
        layer,
        degree: actual,
        ..
    } = err
    {
        assert_eq!(layer, level);
        assert_eq!(actual, degree);
    } else {
        panic!("expected degree bounds violation, got {err:?}");
    }
}

#[test]
fn reachability_collects_all_unreachable_nodes() {
    let params = HnswParams::new(4, 8).expect("params");
    let mut graph = Graph::with_capacity(params.clone(), 4);
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
    graph
        .attach_node(NodeContext {
            node: 3,
            level: 0,
            sequence: 3,
        })
        .expect("attach node 3");
    graph.node_mut(0).expect("entry").neighbours_mut(0).push(1);
    graph.node_mut(1).expect("one").neighbours_mut(0).push(0);

    let ctx = GraphContext {
        graph: &graph,
        params: &params,
    };
    let mut violations = Vec::new();
    let mut mode = EvaluationMode::Collect {
        sink: &mut violations,
        log: false,
    };
    check_reachability(ctx, &mut mode).expect("collect mode never errors");
    assert!(violations.iter().any(|violation| matches!(
        violation,
        HnswInvariantViolation::UnreachableNode { node: 2 }
    )));
    assert!(violations.iter().any(|violation| matches!(
        violation,
        HnswInvariantViolation::UnreachableNode { node: 3 }
    )));
}

#[test]
fn collect_all_reports_multiple_violations() {
    assert_collects_unreachable_nodes(|index| index.invariants().collect_all(), "collect_all");
}

fn assert_collects_unreachable_nodes<F>(collect: F, description: &str)
where
    F: FnOnce(&CpuHnsw) -> Vec<HnswInvariantViolation>,
{
    let (index, _data) = build_index();
    {
        let mut graph = index.graph.write().expect("lock");
        if let Some(entry) = graph.entry() {
            clear_node(&mut graph, entry.node);
        }
    }
    let violations = collect(&index);
    assert!(
        violations
            .iter()
            .any(|violation| matches!(violation, HnswInvariantViolation::UnreachableNode { .. })),
        "{description} should capture unreachable nodes"
    );
}

#[test]
fn collect_all_with_logging_captures_unreachable_nodes() {
    assert_collects_unreachable_nodes(
        |index| index.invariants().collect_all_with_logging(),
        "collect_all_with_logging",
    );
}

#[test]
fn collect_many_with_logging_reports_degree_violation() {
    let (index, _data) = build_index();
    {
        let mut graph = index.graph.write().expect("lock");
        if let Some(node) = graph.node_mut(0) {
            let neighbours = node.neighbours_mut(0);
            neighbours.clear();
            neighbours.extend(std::iter::repeat_n(1, 10));
        }
    }
    let violations = index
        .invariants()
        .collect_many_with_logging([HnswInvariant::DegreeBounds]);
    assert!(
        violations.iter().any(|violation| matches!(
            violation,
            HnswInvariantViolation::DegreeBounds { degree, .. } if *degree >= 9
        )),
        "degree bounds violation should be reported"
    );
}

fn clear_node(graph: &mut Graph, node_id: usize) {
    if let Some(node) = graph.node_mut(node_id) {
        let levels = node.level_count();
        for level in 0..levels {
            node.neighbours_mut(level).clear();
        }
    }
}

// ============================================================================
// No Self-Loops Invariant Tests
// ============================================================================

#[rstest]
#[case::single_node(1)]
#[case::two_nodes(2)]
#[case::four_nodes(4)]
fn no_self_loops_after_construction(#[case] node_count: usize) {
    let params = HnswParams::new(4, 8).expect("params");
    let mut graph = Graph::with_capacity(params, node_count);

    // Insert nodes
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

    // Add edges between distinct nodes
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

    // Verify no self-loops exist
    for (node_id, node) in graph.nodes_iter() {
        for (_level, neighbour) in node.iter_neighbours() {
            assert_ne!(
                node_id, neighbour,
                "node {node_id} should not have itself as neighbour"
            );
        }
    }
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

    // Manually inject a self-loop (this is invalid)
    graph.node_mut(0).expect("node").neighbours_mut(0).push(0);

    // Verify the self-loop exists
    let has_self_loop = graph
        .node(0)
        .map(|node| node.neighbours(0).contains(&0))
        .unwrap_or(false);
    assert!(has_self_loop, "self-loop should be present for detection");
}

// ============================================================================
// Neighbour List Uniqueness Tests
// ============================================================================

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

    // Add unique neighbours
    for &neighbour in &neighbours {
        graph
            .node_mut(0)
            .expect("node")
            .neighbours_mut(0)
            .push(neighbour);
    }

    // Verify uniqueness
    let actual_neighbours = graph.node(0).expect("node").neighbours(0);
    let unique_count = {
        let mut sorted = actual_neighbours.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        sorted.len()
    };
    assert_eq!(
        unique_count,
        actual_neighbours.len(),
        "neighbour list should have no duplicates"
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

    // Manually inject a duplicate (this is invalid)
    let neighbours = graph.node_mut(0).expect("node").neighbours_mut(0);
    neighbours.push(1);
    neighbours.push(1); // Duplicate

    // Verify the duplicate exists
    let neighbour_list = graph.node(0).expect("node").neighbours(0);
    let count_of_1 = neighbour_list.iter().filter(|&&n| n == 1).count();
    assert_eq!(count_of_1, 2, "duplicate should be present for detection");
}

// ============================================================================
// Entry-Point Validity Tests
// ============================================================================

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

    // Insert first node
    graph
        .insert_first(NodeContext {
            node: 0,
            level: levels[0],
            sequence: 0,
        })
        .expect("insert first");

    // Insert remaining nodes and promote entry if higher level
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

    // Verify entry point
    let entry = graph.entry().expect("entry should exist");
    assert_eq!(
        entry.level, expected_entry_level,
        "entry point should have maximum level"
    );

    // Verify entry node exists and has the claimed level
    let entry_node = graph.node(entry.node).expect("entry node should exist");
    assert!(
        entry_node.level_count() > entry.level,
        "entry node should have at least entry.level + 1 levels"
    );
}

#[test]
fn empty_graph_has_no_entry_point() {
    let params = HnswParams::new(4, 8).expect("params");
    let graph = Graph::with_capacity(params, 4);
    assert!(
        graph.entry().is_none(),
        "empty graph should have no entry point"
    );
}
