//! Tests for the CPU HNSW index covering builds, insertions, searches, and
//! error propagation paths.

use super::{CpuHnsw, HnswError, HnswErrorCode, HnswParams, Neighbour};
use crate::{DataSource, DataSourceError};
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
        8 => assert_eq!(forward_ids, vec![0, 1, 2]),
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
        8 => assert_eq!(reverse_ids, vec![2, 1, 0]),
        16 => assert_eq!(reverse_ids, vec![3, 2, 1, 0]),
        _ => unreachable!("unexpected ef in parameterised test"),
    }
    assert_sorted_by_distance(&neighbours);
}

#[rstest]
fn uses_batch_distances_during_scoring() {
    #[derive(Clone)]
    struct InstrumentedSource {
        data: Vec<f32>,
        batch_calls: Arc<AtomicUsize>,
    }

    impl InstrumentedSource {
        fn new(data: Vec<f32>, batch_calls: Arc<AtomicUsize>) -> Self {
            Self { data, batch_calls }
        }
    }

    impl DataSource for InstrumentedSource {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn name(&self) -> &str {
            "instrumented"
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
                        .data
                        .get(query)
                        .ok_or(DataSourceError::OutOfBounds { index: query })?;
                    let b = self
                        .data
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
fn duplicate_insert_is_rejected() {
    let source = DummySource::new(vec![0.0, 1.0, 2.0]);
    let params = HnswParams::new(2, 4)
        .expect("params must be valid")
        .with_rng_seed(7);
    let index = CpuHnsw::build(&source, params).expect("initial build must succeed");
    let err = index.insert(0, &source).expect_err("duplicate must fail");
    assert!(matches!(err, HnswError::DuplicateNode { node: 0 }));
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

    let params = HnswParams::new(2, 4).expect("params must be valid");
    let err = CpuHnsw::build(&BatchNan, params).expect_err("build must fail on batch NaN");
    match err {
        HnswError::NonFiniteDistance { left, right } => {
            assert_eq!(left, 2);
            assert_eq!(right, 1);
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
        let [left, right]: &[Neighbour; 2] =
            window.try_into().expect("windows(2) always yields pairs");
        assert!(
            left.distance <= right.distance + f32::EPSILON,
            "distances must be non-decreasing: {neighbours:?}",
        );
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
