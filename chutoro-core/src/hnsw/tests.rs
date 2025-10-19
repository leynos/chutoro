use super::{CpuHnsw, HnswError, HnswParams, Neighbour};
use crate::{DataSource, DataSourceError};
use rstest::rstest;
use std::num::NonZeroUsize;

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

fn contains_id(neighbours: &[Neighbour], id: usize) -> bool {
    neighbours.iter().any(|neighbour| neighbour.id == id)
}
