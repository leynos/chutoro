use super::{
    EvaluationMode, GraphContext, HnswInvariantViolation, check_degree_bounds, check_reachability,
    helpers,
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
    let mut mode = EvaluationMode::Collect(&mut violations);
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
    let (index, _data) = build_index();
    {
        let mut graph = index.graph.write().expect("lock");
        if let Some(entry) = graph.entry() {
            clear_node(&mut graph, entry.node);
        }
    }
    let violations = index.invariants().collect_all();
    assert!(
        violations
            .iter()
            .any(|violation| matches!(violation, HnswInvariantViolation::UnreachableNode { .. })),
        "collect_all should capture unreachable nodes"
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
