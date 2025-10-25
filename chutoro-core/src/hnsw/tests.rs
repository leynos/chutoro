//! Tests for the CPU HNSW index covering builds, insertions, searches, and
//! error propagation paths.

use super::{
    CpuHnsw, HnswError, HnswErrorCode, HnswParams, Neighbour,
    graph::{EdgeContext, Graph, SearchContext},
    insert::TrimJob,
};
use crate::{DataSource, DataSourceError, test_utils::CountingSource};
use rand::{Rng, SeedableRng, distributions::Standard, rngs::SmallRng};
use rstest::rstest;
use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Clone)]
struct DummySource {
    data: Vec<f32>,
}

impl DummySource {
    fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl DataSource for DummySource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        "dummy"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let a = self
            .data
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let b = self
            .data
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok((a - b).abs())
    }
}

#[rstest]
#[case(2, 8)]
#[case(4, 16)]
fn builds_and_searches(#[case] m: usize, #[case] ef: usize) {
    let source = DummySource::new(vec![0.0, 1.0, 2.0, 5.0]);
    let params = HnswParams::new(m, ef)
        .expect("params must be valid")
        .with_rng_seed(42);
    let index = CpuHnsw::build(&source, params).expect("build must succeed");

    let neighbours = index
        .search(
            &source,
            0,
            NonZeroUsize::new(ef).expect("ef must be non-zero"),
        )
        .expect("search must succeed");
    let forward_ids: Vec<_> = neighbours.iter().map(|n| n.id).collect();
    match ef {
        8 => {
            assert!(forward_ids.starts_with(&[0, 1, 2]));
            if forward_ids.len() == 4 {
                assert_eq!(forward_ids[3], 3);
            } else {
                assert_eq!(forward_ids.len(), 3);
            }
        }
        16 => assert_eq!(forward_ids, vec![0, 1, 2, 3]),
        _ => unreachable!("unexpected ef in parameterised test"),
    }
    assert_sorted_by_distance(&neighbours);

    let neighbours = index
        .search(
            &source,
            3,
            NonZeroUsize::new(ef).expect("ef must be non-zero"),
        )
        .expect("search must succeed");
    let reverse_ids: Vec<_> = neighbours.iter().map(|n| n.id).collect();
    match ef {
        8 => {
            assert!(reverse_ids.ends_with(&[2, 1, 0]));
            if reverse_ids.len() == 4 {
                assert_eq!(reverse_ids[0], 3);
            } else {
                assert_eq!(reverse_ids.len(), 3);
            }
        }
        16 => assert_eq!(reverse_ids, vec![3, 2, 1, 0]),
        _ => unreachable!("unexpected ef in parameterised test"),
    }
    assert_sorted_by_distance(&neighbours);
}

#[rstest]
fn uses_batch_distances_during_scoring() {
    #[derive(Clone)]
    struct InstrumentedSource {
        base: CountingSource,
        batch_calls: Arc<AtomicUsize>,
    }

    impl InstrumentedSource {
        fn new(data: Vec<f32>, batch_calls: Arc<AtomicUsize>) -> Self {
            let base =
                CountingSource::with_name("instrumented", data, Arc::new(AtomicUsize::new(0)));
            Self { base, batch_calls }
        }
    }

    impl DataSource for InstrumentedSource {
        fn len(&self) -> usize {
            self.base.len()
        }

        fn name(&self) -> &str {
            self.base.name()
        }

        fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
            self.base.distance(left, right)
        }

        fn batch_distances(
            &self,
            query: usize,
            candidates: &[usize],
        ) -> Result<Vec<f32>, DataSourceError> {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            candidates
                .iter()
                .map(|&candidate| {
                    let a = self
                        .base
                        .data()
                        .get(query)
                        .ok_or(DataSourceError::OutOfBounds { index: query })?;
                    let b = self
                        .base
                        .data()
                        .get(candidate)
                        .ok_or(DataSourceError::OutOfBounds { index: candidate })?;
                    Ok((a - b).abs())
                })
                .collect()
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let source = InstrumentedSource::new(vec![0.0, 1.0, 2.0, 5.0], Arc::clone(&calls));
    let params = HnswParams::new(2, 4)
        .expect("params must be valid")
        .with_rng_seed(11);
    let index = CpuHnsw::build(&source, params).expect("build must succeed");

    index
        .search(
            &source,
            1,
            NonZeroUsize::new(4).expect("ef must be non-zero"),
        )
        .expect("search must succeed");

    assert!(
        calls.load(Ordering::Relaxed) > 0,
        "batch distances should be exercised"
    );
}

#[rstest]
fn greedy_descent_selects_closest_neighbour() {
    let source = DummySource::new(vec![1.0, 0.8, 0.6, 0.0]);
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, source.len());
    graph.attach_node(0, 0).expect("attach entry");
    graph.attach_node(1, 0).expect("attach neighbour one");
    graph.attach_node(2, 0).expect("attach neighbour two");
    graph.attach_node(3, 0).expect("attach query node");
    graph
        .node_mut(0)
        .expect("entry must exist")
        .neighbours_mut(0)
        .extend([1, 2]);

    let ctx = SearchContext {
        query: 3,
        entry: 0,
        level: 0,
    };
    let result = graph
        .searcher()
        .greedy_search_layer(&source, ctx)
        .expect("greedy search must succeed");

    assert_eq!(
        result, 2,
        "greedy descent should pick the closest neighbour",
    );
}

#[test]
fn layer_search_explores_equal_distance_candidates() {
    let source = DummySource::new(vec![0.0, 1.0, 1.0, 0.2]);
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, source.len());

    graph.insert_first(1, 0).expect("seed entry point");
    graph.attach_node(0, 0).expect("attach query node");
    graph.attach_node(2, 0).expect("attach tie candidate");
    graph.attach_node(3, 0).expect("attach hidden closer node");

    graph
        .node_mut(1)
        .expect("entry must exist")
        .neighbours_mut(0)
        .extend([2]);
    graph
        .node_mut(2)
        .expect("tie candidate must exist")
        .neighbours_mut(0)
        .extend([1, 3]);
    graph
        .node_mut(3)
        .expect("closer node must exist")
        .neighbours_mut(0)
        .extend([2]);

    let searcher = graph.searcher();
    let ctx = SearchContext {
        query: 0,
        entry: 1,
        level: 0,
    }
    .with_ef(1);

    let neighbours = searcher
        .search_layer(&source, ctx)
        .expect("layer search must succeed");

    assert_eq!(neighbours.len(), 1, "ef=1 should cap the result set");
    let neighbour = &neighbours[0];
    assert_eq!(neighbour.id, 3, "search should reach the closer node");
    assert!(
        neighbour.distance < 1.0,
        "closer node must improve the bound"
    );
}

#[test]
fn search_respects_minimum_ef() {
    let source = DummySource::new(vec![0.0, 1.5, 3.0]);
    let params = HnswParams::new(2, 4)
        .expect("params must be valid")
        .with_rng_seed(29);
    let index = CpuHnsw::build(&source, params).expect("build must succeed");

    let neighbours = index
        .search(
            &source,
            0,
            NonZeroUsize::new(1).expect("ef must be non-zero"),
        )
        .expect("search must succeed");

    assert_eq!(neighbours.len(), 1, "ef=1 must return a single result");
    assert_eq!(neighbours[0].id, 0, "search should favour the closest node");
}

#[rstest]
fn duplicate_insert_is_rejected() {
    let source = DummySource::new(vec![0.0, 1.0, 2.0]);
    let params = HnswParams::new(2, 4)
        .expect("params must be valid")
        .with_rng_seed(7);
    let index = CpuHnsw::build(&source, params).expect("initial build must succeed");
    let err = index.insert(0, &source).expect_err("duplicate must fail");
    assert!(matches!(err, HnswError::DuplicateNode { node: 0 }));
}

#[test]
fn cpu_hnsw_initialises_graph_with_params() -> Result<(), HnswError> {
    let params = HnswParams::new(2, 4)?;
    let index = CpuHnsw::with_capacity(params.clone(), 8)?;

    let (propagated, has_last_slot) =
        index.inspect_graph(|graph| (graph.params().clone(), graph.has_slot(7)));

    assert_eq!(
        propagated.max_connections(),
        params.max_connections(),
        "graph should retain neighbour fan-out",
    );
    assert_eq!(
        propagated.ef_construction(),
        params.ef_construction(),
        "graph should retain ef_construction",
    );
    assert_eq!(
        propagated.max_level(),
        params.max_level(),
        "graph should retain max level",
    );
    assert_eq!(
        propagated.rng_seed(),
        params.rng_seed(),
        "graph should retain RNG seed",
    );
    assert!(has_last_slot, "graph capacity must expose final slot");
    Ok(())
}

#[test]
fn trimming_keeps_new_node_on_distance_ties() -> Result<(), HnswError> {
    let params = HnswParams::new(1, 4)?;
    let index = CpuHnsw::with_capacity(params.clone(), 3)?;
    let ctx = EdgeContext {
        level: 0,
        max_connections: params.max_connections(),
    };
    let job = TrimJob {
        node: 0,
        ctx,
        candidates: vec![2, 1],
    };

    let results = index
        .score_trim_jobs(vec![job], &DummySource::new(vec![0.0, 1.0, 1.0]))?
        .into_iter()
        .next()
        .expect("trim job yields a result");

    assert_eq!(
        results.neighbours,
        vec![2],
        "stable sorting must retain the new node when distances tie",
    );
    Ok(())
}

#[rstest]
fn non_finite_distance_is_reported() {
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

    let params = HnswParams::new(2, 4).expect("params must be valid");
    let err = CpuHnsw::build(&NanSource, params).expect_err("build must fail on NaN");
    match err {
        HnswError::NonFiniteDistance { left, right } => {
            assert_eq!(left, 0);
            assert_eq!(right, 0);
        }
        other => panic!("expected non-finite distance, got {other:?}"),
    }
}

#[test]
fn reports_invariant_violation_when_search_node_missing() {
    let source = DummySource::new(vec![0.0, 1.0]);
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let mut graph = Graph::with_capacity(params, 2);
    graph
        .insert_first(0, 0)
        .expect("initial node must insert successfully");
    graph
        .node_mut(0)
        .expect("node 0 must exist")
        .neighbours_mut(0)
        .push(1);

    let ctx = SearchContext {
        query: 0,
        entry: 0,
        level: 0,
    }
    .with_ef(2);

    let err = graph
        .searcher()
        .search_layer(&source, ctx)
        .expect_err("missing node must surface an invariant violation");
    match err {
        HnswError::GraphInvariantViolation { message } => {
            assert_eq!(message, "node 1 missing during layer search at level 0");
        }
        other => panic!("expected GraphInvariantViolation, got {other:?}"),
    }
}

#[test]
fn attach_node_rejects_excessive_levels() {
    let params = HnswParams::new(2, 4)
        .expect("params must be valid")
        .with_max_level(1);
    let mut graph = Graph::with_capacity(params, 3);
    graph.attach_node(0, 1).expect("level within bounds");

    let err = graph
        .attach_node(1, 3)
        .expect_err("level above max must fail");
    match err {
        HnswError::InvalidParameters { reason } => {
            assert!(
                reason.contains("exceeds max_level"),
                "error should report level overflow: {reason}"
            );
        }
        other => panic!("expected InvalidParameters, got {other:?}"),
    }
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

    let params = HnswParams::new(1, 1).expect("params must be valid");
    let err = CpuHnsw::build(&BatchNan, params).expect_err("build must fail on batch NaN");
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
                involves_initial_pair(0, other) || involves_initial_pair(1, other),
                "unexpected counterpart for non-finite edge: {other}",
            );
        }
    }

    match err {
        HnswError::NonFiniteDistance { left, right } => {
            validate_non_finite_participants(left, right);
        }
        other => panic!("expected non-finite distance, got {other:?}"),
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
fn accepts_equal_search_and_connection_width() {
    let params = HnswParams::new(8, 8).expect("equal widths must be valid");
    assert_eq!(params.max_connections(), 8);
    assert_eq!(params.ef_construction(), 8);
}

#[test]
fn with_capacity_rejects_zero_capacity() {
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let err = CpuHnsw::with_capacity(params, 0).expect_err("capacity must be positive");
    assert!(matches!(err, HnswError::InvalidParameters { .. }));
}

#[test]
fn level_sampling_matches_geometric_tail() {
    let params = HnswParams::new(16, 64)
        .expect("params must be valid")
        .with_rng_seed(1337);
    let mut rng = SmallRng::seed_from_u64(params.rng_seed());
    let mut counts = vec![0_usize; params.max_level() + 1];
    let samples = 10_000;
    for _ in 0..samples {
        let mut level = 0_usize;
        while level < params.max_level() {
            let draw: f64 = rng.sample(Standard);
            if params.should_stop(draw) {
                break;
            }
            level += 1;
        }
        counts[level] += 1;
    }

    let continue_prob = 1.0 / params.max_connections() as f64;
    for window in counts
        .windows(2)
        .filter(|pair| pair[0] > 0 && pair[1] > 0)
        .take(3)
    {
        let next_ratio = window[1] as f64 / window[0] as f64;
        assert!(
            (next_ratio - continue_prob).abs() < 0.035,
            "ratio should approach geometric tail (observed {next_ratio}, expected {continue_prob})",
        );
    }
}

fn assert_sorted_by_distance(neighbours: &[Neighbour]) {
    for window in neighbours.windows(2) {
        if let [left, right] = window {
            assert!(
                left.distance <= right.distance + f32::EPSILON,
                "distances must be non-decreasing: {neighbours:?}",
            );
        }
    }
}

fn new_node_counterpart(left: usize, right: usize, new_node: usize) -> Option<usize> {
    if left == new_node {
        Some(right)
    } else if right == new_node {
        Some(left)
    } else {
        None
    }
}

#[test]
fn exposes_machine_readable_error_codes() {
    assert_eq!(HnswError::EmptyBuild.code(), HnswErrorCode::EmptyBuild);
    assert_eq!(
        HnswError::InvalidParameters {
            reason: "bad".into(),
        }
        .code(),
        HnswErrorCode::InvalidParameters,
    );
    assert_eq!(
        HnswError::DuplicateNode { node: 3 }.code(),
        HnswErrorCode::DuplicateNode,
    );
    assert_eq!(HnswError::GraphEmpty.code(), HnswErrorCode::GraphEmpty);
    assert_eq!(
        HnswError::GraphInvariantViolation {
            message: "oops".into(),
        }
        .code(),
        HnswErrorCode::GraphInvariantViolation,
    );
    assert_eq!(
        HnswError::NonFiniteDistance { left: 0, right: 1 }.code(),
        HnswErrorCode::NonFiniteDistance,
    );
    assert_eq!(
        HnswError::from(DataSourceError::EmptyData).code(),
        HnswErrorCode::DataSource,
    );
    assert_eq!(HnswErrorCode::DataSource.as_str(), "DATA_SOURCE");
}
