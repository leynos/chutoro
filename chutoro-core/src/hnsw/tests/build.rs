//! Build and end-to-end search tests for the CPU HNSW index.

use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use rstest::rstest;

use crate::{
    DataSource, DataSourceError,
    hnsw::{CpuHnsw, HnswError, HnswParams, graph::EdgeContext, insert::TrimJob},
    test_utils::CountingSource,
};

use super::fixtures::{DummySource, assert_sorted_by_distance};

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
        "batch distances should be exercised",
    );
}

#[rstest]
fn duplicate_insert_is_rejected() {
    let source = DummySource::new(vec![0.0, 1.0, 2.0]);
    let params = HnswParams::new(2, 4).expect("params must be valid");
    let index =
        CpuHnsw::with_capacity(params.clone(), source.len()).expect("index should allocate");
    index.insert(0, &source).expect("first insert succeeds");

    let err = index
        .insert(0, &source)
        .expect_err("duplicate insert fails");
    assert!(matches!(err, HnswError::DuplicateNode { node: 0 }));
}

#[rstest]
fn cpu_hnsw_initialises_graph_with_params() -> Result<(), HnswError> {
    let params = HnswParams::new(2, 4)?.with_rng_seed(7);
    let index = CpuHnsw::with_capacity(params.clone(), 8)?;
    index.inspect_graph(|graph| {
        let graph_params = graph.params();
        assert_eq!(graph_params.max_connections(), params.max_connections());
        assert_eq!(graph_params.ef_construction(), params.ef_construction());
        assert_eq!(graph_params.rng_seed(), params.rng_seed());
    });
    Ok(())
}

#[rstest]
fn trimming_prefers_lower_id_on_distance_ties() -> Result<(), HnswError> {
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
        sequences: vec![2_u64, 1_u64],
    };

    let result = index
        .score_trim_jobs(vec![job], &DummySource::new(vec![0.0, 1.0, 1.0]))?
        .into_iter()
        .next()
        .expect("trim job yields a result");

    assert_eq!(
        result.neighbours,
        vec![1],
        "deterministic tie-breaking must prefer the lower identifier",
    );
    Ok(())
}
