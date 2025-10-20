use super::{CpuHnsw, HnswError, HnswErrorCode, HnswParams, Neighbour};
use crate::{DataSource, DataSourceError};
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
    assert!(contains_id(&neighbours, 1));

    let neighbours = index
        .search(
            &source,
            3,
            NonZeroUsize::new(ef).expect("ef must be non-zero"),
        )
        .expect("search must succeed");
    assert!(contains_id(&neighbours, 2));
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
    assert!(matches!(err, HnswError::NonFiniteDistance { .. }));
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
    assert!(matches!(err, HnswError::NonFiniteDistance { .. }));
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

fn contains_id(neighbours: &[Neighbour], id: usize) -> bool {
    neighbours.iter().any(|neighbour| neighbour.id == id)
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
