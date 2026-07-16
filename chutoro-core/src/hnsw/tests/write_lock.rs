//! Write-lock invariant tests for HNSW distance scoring.

use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use proptest::{
    collection::vec as prop_vec,
    prelude::*,
    test_runner::{TestCaseError, TestCaseResult},
};
use rstest::fixture;
use rstest::rstest;

use crate::{
    DataSource, DataSourceError,
    hnsw::{CpuHnsw, HnswError, HnswParams},
    test_utils::CountingSource,
};

use super::super::cpu::test_helpers::WriteGraphMarkerGuard;

const WRITE_LOCK_SCORING_PANIC: &str =
    "distance scoring must not run while the current thread holds the HNSW write graph lock";

#[derive(Clone)]
struct WriteLockAssertingSource {
    base: CountingSource,
    scoring_calls: Arc<AtomicUsize>,
}

impl WriteLockAssertingSource {
    fn new(data: Vec<f32>) -> Self {
        Self {
            base: CountingSource::with_name(
                "write-lock-asserting",
                data,
                Arc::new(AtomicUsize::new(0)),
            ),
            scoring_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn scoring_calls(&self) -> usize {
        self.scoring_calls.load(Ordering::Relaxed)
    }

    fn assert_not_writing_graph(&self) {
        assert!(
            !CpuHnsw::current_thread_holds_write_graph_for_test(),
            "{WRITE_LOCK_SCORING_PANIC}",
        );
        self.scoring_calls.fetch_add(1, Ordering::Relaxed);
    }

    fn distance_value(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let a = self
            .base
            .data()
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let b = self
            .base
            .data()
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok((a - b).abs())
    }
}

impl DataSource for WriteLockAssertingSource {
    fn len(&self) -> usize {
        self.base.len()
    }

    fn name(&self) -> &str {
        self.base.name()
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        self.assert_not_writing_graph();
        self.distance_value(left, right)
    }

    fn batch_distances(
        &self,
        query: usize,
        candidates: &[usize],
    ) -> Result<Vec<f32>, DataSourceError> {
        self.assert_not_writing_graph();
        candidates
            .iter()
            .map(|&candidate| self.distance_value(query, candidate))
            .collect()
    }

    fn distance_batch(
        &self,
        pairs: &[(usize, usize)],
        out: &mut [f32],
    ) -> Result<(), DataSourceError> {
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }

        self.assert_not_writing_graph();
        for ((left, right), slot) in pairs.iter().copied().zip(out.iter_mut()) {
            *slot = self.distance_value(left, right)?;
        }
        Ok(())
    }
}

struct WriteLockScenario {
    source: WriteLockAssertingSource,
    index: CpuHnsw,
}

#[derive(Debug)]
struct WriteLockPropertyCase {
    values: Vec<f32>,
    max_connections: usize,
    ef_construction: usize,
    rng_seed: u64,
    query: usize,
    search_ef: usize,
}

#[fixture]
fn write_graph_marker_guard() -> WriteGraphMarkerGuard {
    CpuHnsw::enable_write_graph_marker_for_test()
}

#[fixture]
fn write_lock_source() -> WriteLockAssertingSource {
    WriteLockAssertingSource::new(vec![0.0, 1.0, 2.0, 5.0, 8.0, 13.0])
}

#[fixture]
fn write_lock_params() -> HnswParams {
    HnswParams::new(2, 4)
        .expect("params must be valid")
        .with_rng_seed(11)
}

#[fixture]
fn built_write_lock_scenario(
    write_lock_source: WriteLockAssertingSource,
    write_lock_params: HnswParams,
) -> WriteLockScenario {
    let index = CpuHnsw::build(&write_lock_source, write_lock_params).expect("build must succeed");
    WriteLockScenario {
        source: write_lock_source,
        index,
    }
}

fn write_lock_case_strategy() -> impl Strategy<Value = WriteLockPropertyCase> {
    (2_usize..=12).prop_flat_map(|len| {
        let max_connections = (len - 1).clamp(1, 4);
        (
            prop_vec(-1000_i16..=1000, len),
            1_usize..=max_connections,
            0_usize..=8,
            any::<u64>(),
            0_usize..len,
            1_usize..=len,
        )
            .prop_map(
                move |(raw_values, max_connections, ef_extra, rng_seed, query, search_ef)| {
                    WriteLockPropertyCase {
                        values: insertion_ordered_values(raw_values),
                        max_connections,
                        ef_construction: max_connections + ef_extra,
                        rng_seed,
                        query,
                        search_ef,
                    }
                },
            )
    })
}

fn insertion_ordered_values(raw_values: Vec<i16>) -> Vec<f32> {
    raw_values
        .into_iter()
        .enumerate()
        .map(|(index, value)| f32::from(value) + index as f32 / 1024.0)
        .collect()
}

fn assert_write_lock_property(case: WriteLockPropertyCase) -> TestCaseResult {
    let _marker_guard = CpuHnsw::enable_write_graph_marker_for_test();
    let source = WriteLockAssertingSource::new(case.values);
    let params = HnswParams::new(case.max_connections, case.ef_construction)
        .map_err(|error| TestCaseError::fail(format!("invalid generated params: {error}")))?
        .with_rng_seed(case.rng_seed);
    let index = build_generated_index(&source, params)
        .map_err(|error| TestCaseError::fail(format!("generated build failed: {error}")))?;

    index
        .search(
            &source,
            case.query,
            NonZeroUsize::new(case.search_ef).expect("generated ef must be non-zero"),
        )
        .map_err(|error| TestCaseError::fail(format!("generated search failed: {error}")))?;

    prop_assert!(
        source.scoring_calls() > 0,
        "the marker should observe scoring for generated cases",
    );
    prop_assert!(!CpuHnsw::current_thread_holds_write_graph_for_test());
    Ok(())
}

fn build_generated_index(
    source: &WriteLockAssertingSource,
    params: HnswParams,
) -> Result<CpuHnsw, HnswError> {
    let index = CpuHnsw::with_capacity(params, source.len())?;
    for node in 0..source.len() {
        index.insert(node, source)?;
    }
    Ok(index)
}

#[rstest]
fn write_graph_marker_is_scoped_to_the_current_thread(
    _write_graph_marker_guard: WriteGraphMarkerGuard,
    write_lock_params: HnswParams,
) -> Result<(), HnswError> {
    let index = CpuHnsw::with_capacity(write_lock_params, 2).expect("index should allocate");
    assert!(!CpuHnsw::current_thread_holds_write_graph_for_test());
    index.write_graph(|_graph| {
        assert!(CpuHnsw::current_thread_holds_write_graph_for_test());
        Ok(())
    })?;
    assert!(!CpuHnsw::current_thread_holds_write_graph_for_test());
    Ok(())
}

#[rstest]
fn hnsw_scoring_does_not_run_inside_write_graph_scope(
    _write_graph_marker_guard: WriteGraphMarkerGuard,
    built_write_lock_scenario: WriteLockScenario,
) {
    built_write_lock_scenario
        .index
        .search(
            &built_write_lock_scenario.source,
            1,
            NonZeroUsize::new(4).expect("ef must be non-zero"),
        )
        .expect("search must succeed");

    assert!(
        built_write_lock_scenario.source.scoring_calls() > 0,
        "the guard should observe real HNSW scoring calls",
    );
}

#[rstest]
#[should_panic(expected = "distance scoring must not run while the current thread holds")]
fn write_lock_scoring_guard_has_teeth(
    _write_graph_marker_guard: WriteGraphMarkerGuard,
    write_lock_source: WriteLockAssertingSource,
    write_lock_params: HnswParams,
) {
    let index = CpuHnsw::with_capacity(write_lock_params, 2).expect("index should allocate");
    index
        .write_graph(|_graph| {
            let _distances = write_lock_source
                .batch_distances(0, &[1])
                .expect("inside-lock scoring call should otherwise succeed");
            Ok(())
        })
        .expect("inside-lock scoring should panic before returning");
}

proptest! {
    #![proptest_config(crate::test_utils::suite_proptest_config(16))]

    #[test]
    fn generated_hnsw_scoring_does_not_run_inside_write_graph_scope(
        case in write_lock_case_strategy()
    ) {
        assert_write_lock_property(case)?;
    }
}
