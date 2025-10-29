//! Error handling tests covering invalid parameters and data source failures.

use rstest::rstest;

use crate::{
    DataSource, DataSourceError,
    hnsw::{
        CpuHnsw, HnswError, HnswParams,
        distance_cache::{DistanceCache, DistanceCacheConfig},
        graph::{Graph, NodeContext, SearchContext},
        validate::validate_batch_distances,
    },
};

use super::fixtures::{DummySource, new_node_counterpart};

#[rstest]
fn non_finite_distance_is_reported() {
    #[derive(Clone)]
    struct NanSource;

    impl DataSource for NanSource {
        fn len(&self) -> usize {
            2
        }

        fn name(&self) -> &str {
            "nan"
        }

        fn distance(&self, _: usize, _: usize) -> Result<f32, DataSourceError> {
            Ok(f32::NAN)
        }
    }

    let params = HnswParams::new(1, 2).expect("params must be valid");
    let err = CpuHnsw::build(&NanSource, params).expect_err("build must fail on NaN");
    match err {
        HnswError::NonFiniteDistance { .. } => {}
        other => panic!("expected non-finite distance, got {other:?}"),
    }
}

#[rstest]
fn reports_invariant_violation_when_search_node_missing() {
    let source = DummySource::new(vec![0.0, 1.0, 2.0]);
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, source.len());
    graph
        .insert_first(NodeContext {
            node: 0,
            level: 0,
            sequence: 0,
        })
        .expect("initial node must insert successfully");
    graph
        .node_mut(0)
        .expect("entry exists")
        .neighbours_mut(0)
        .push(1);

    let searcher = graph.searcher();
    let ctx = SearchContext {
        query: 0,
        entry: 0,
        level: 0,
    }
    .with_ef(3);

    let err = searcher
        .search_layer(None, &source, ctx)
        .expect_err("missing node must surface an invariant violation");

    match err {
        HnswError::GraphInvariantViolation { message } => {
            assert_eq!(
                message,
                "sequence missing for node 1 during layer expansion"
            );
        }
        other => panic!("expected invariant violation, got {other:?}"),
    }
}

#[rstest]
fn attach_node_rejects_excessive_levels() {
    let params = HnswParams::new(1, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params.clone(), 2);
    let err = graph
        .attach_node(NodeContext {
            node: 0,
            level: params.max_level() + 1,
            sequence: 0,
        })
        .expect_err("excessive level must be rejected");
    assert!(matches!(err, HnswError::InvalidParameters { .. }));
}

#[rstest]
fn non_finite_batch_distance_is_reported() {
    #[derive(Clone, Copy)]
    struct BatchNan;

    impl DataSource for BatchNan {
        fn len(&self) -> usize {
            3
        }

        fn name(&self) -> &str {
            "batch-nan"
        }

        fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
            Ok((left as f32 - right as f32).abs())
        }

        fn batch_distances(
            &self,
            _: usize,
            candidates: &[usize],
        ) -> Result<Vec<f32>, DataSourceError> {
            Ok(vec![f32::NAN; candidates.len()])
        }
    }

    let cache = DistanceCache::new(DistanceCacheConfig::default());
    let err = validate_batch_distances(Some(&cache), &BatchNan, 0, &[1, 2])
        .expect_err("batch distance validation must reject NaNs");

    fn involves_new_node(left: usize, right: usize) -> bool {
        left == 2 || right == 2
    }

    fn involves_initial_pair(left: usize, right: usize) -> bool {
        matches!((left, right), (0, 1) | (1, 0))
    }

    fn validate_non_finite_participants(left: usize, right: usize) {
        assert!(
            involves_new_node(left, right) || involves_initial_pair(left, right),
            "unexpected participants for non-finite edge: ({left}, {right})",
        );

        if involves_new_node(left, right) {
            let other =
                new_node_counterpart(left, right, 2).expect("new node counterpart must exist");
            assert!(
                matches!(other, 0 | 1),
                "unexpected counterpart for non-finite edge: {other}",
            );
        }
    }

    if let HnswError::NonFiniteDistance { left, right } = err {
        validate_non_finite_participants(left, right);
    } else {
        panic!("expected non-finite distance, got {err:?}");
    }
}

#[rstest]
fn rejects_invalid_parameters(#[values(0, 3)] max_connections: usize) {
    if max_connections == 0 {
        let err = HnswParams::new(0, 4).expect_err("zero connections invalid");
        assert!(matches!(err, HnswError::InvalidParameters { .. }));
    } else {
        let err = HnswParams::new(max_connections, max_connections - 1)
            .expect_err("ef must exceed connections");
        assert!(matches!(err, HnswError::InvalidParameters { .. }));
    }
}

#[test]
fn with_capacity_rejects_zero_capacity() {
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let err = CpuHnsw::with_capacity(params, 0).expect_err("capacity must be positive");
    assert!(matches!(err, HnswError::InvalidParameters { .. }));
}
