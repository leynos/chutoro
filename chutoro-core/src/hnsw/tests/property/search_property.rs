//! Search correctness helpers for the CPU HNSW property suite.
//!
//! Hosts the brute-force oracle, recall measurement utilities, configurable
//! recall threshold parsing, and targeted unit tests that exercise the helper
//! logic directly via `rstest`.

use std::{
    collections::HashSet,
    env,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use proptest::test_runner::{TestCaseError, TestCaseResult};
use rstest::rstest;

use super::types::HnswFixture;
use crate::error::DataSourceError;
use crate::{CpuHnsw, DataSource, Neighbour};

/// Executes the search-correctness property for a generated fixture.
pub(super) fn run_search_correctness_property(
    fixture: HnswFixture,
    query_hint: u16,
    k_hint: u16,
) -> TestCaseResult {
    let config = SearchPropertyConfig::load();
    let params = fixture
        .params
        .build()
        .map_err(|err| TestCaseError::fail(format!("invalid params: {err}")))?;
    let source = fixture
        .clone()
        .into_source()
        .map_err(|err| TestCaseError::fail(format!("fixture -> source failed: {err}")))?;

    let len = source.len();
    if len < 2 {
        return Ok(());
    }

    const MIN_VALID_CONNECTIONS: usize = 16;
    if fixture.params.max_connections < MIN_VALID_CONNECTIONS {
        return Ok(());
    }
    let query = (usize::from(query_hint) % len).min(len.saturating_sub(1));
    let fanout_cap = fixture.params.max_connections.max(2);
    let max_k = len.min(16).min(fanout_cap);
    let k = ((usize::from(k_hint) % max_k).max(1)).min(len);
    let ef = NonZeroUsize::new(len.max(k * 2).max(16)).expect("ef must be non-zero");

    let index = CpuHnsw::build(&source, params)
        .map_err(|err| TestCaseError::fail(format!("build failed: {err}")))?;

    let search_started = Instant::now();
    let mut hnsw_neighbours = index
        .search(&source, query, ef)
        .map_err(|err| TestCaseError::fail(format!("search failed: {err}")))?;
    let hnsw_elapsed = search_started.elapsed();

    hnsw_neighbours.truncate(k);

    let oracle_started = Instant::now();
    let oracle = brute_force_top_k(&source, query, k)
        .map_err(|err| TestCaseError::fail(format!("oracle failed: {err}")))?;
    let oracle_elapsed = oracle_started.elapsed();

    let recall = recall_at_k(&oracle, &hnsw_neighbours, k);
    record_search_metrics(SearchMetricsContext {
        fixture: &fixture,
        len,
        k,
        recall,
        threshold: config.min_recall(),
        timings: SearchTimings::new(hnsw_elapsed, oracle_elapsed),
    });

    if recall < config.min_recall() {
        return Err(TestCaseError::fail(format!(
            "recall {recall:.3} below threshold {threshold:.3} (len={len}, k={k}, query={query}, max_connections={}, fanout_cap={}, distribution={:?})",
            fixture.params.max_connections,
            fanout_cap,
            fixture.distribution,
            threshold = config.min_recall(),
        )));
    }

    Ok(())
}

fn brute_force_top_k<D: DataSource + Sync>(
    source: &D,
    query: usize,
    k: usize,
) -> Result<Vec<Neighbour>, DataSourceError> {
    let mut neighbours: Vec<Neighbour> = (0..source.len())
        .map(|candidate| -> Result<Neighbour, DataSourceError> {
            let distance = source.distance(query, candidate)?;
            Ok(Neighbour {
                id: candidate,
                distance,
            })
        })
        .collect::<Result<_, _>>()?;
    neighbours.sort_unstable();
    neighbours.truncate(k);
    Ok(neighbours)
}

fn recall_at_k(oracle: &[Neighbour], observed: &[Neighbour], k: usize) -> f32 {
    assert!(k > 0, "k must be positive");
    if oracle.is_empty() {
        return 0.0;
    }
    let target = k.min(oracle.len());
    let oracle_ids: HashSet<usize> = oracle.iter().take(target).map(|n| n.id).collect();
    let hits = observed
        .iter()
        .take(target)
        .filter(|neighbour| oracle_ids.contains(&neighbour.id))
        .count();
    hits as f32 / target as f32
}

#[derive(Clone, Copy, Debug)]
struct SearchTimings {
    hnsw: Duration,
    oracle: Duration,
}

impl SearchTimings {
    fn new(hnsw: Duration, oracle: Duration) -> Self {
        Self { hnsw, oracle }
    }

    fn speedup(&self) -> f64 {
        let oracle_secs = self.oracle.as_secs_f64();
        let hnsw_secs = self.hnsw.as_secs_f64().max(f64::EPSILON);
        if oracle_secs <= f64::EPSILON {
            f64::INFINITY
        } else {
            oracle_secs / hnsw_secs
        }
    }
}

struct SearchMetricsContext<'a> {
    fixture: &'a HnswFixture,
    len: usize,
    k: usize,
    recall: f32,
    threshold: f32,
    timings: SearchTimings,
}

fn record_search_metrics(ctx: SearchMetricsContext<'_>) {
    tracing::info!(
        distribution = ?ctx.fixture.distribution,
        dimension = ctx.fixture.dimension(),
        len = ctx.len,
        k = ctx.k,
        recall = ctx.recall,
        threshold = ctx.threshold,
        hnsw_micros = ctx.timings.hnsw.as_secs_f64() * 1_000_000.0,
        oracle_micros = ctx.timings.oracle.as_secs_f64() * 1_000_000.0,
        speedup = ctx.timings.speedup(),
        "hnsw search correctness property",
    );
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RecallThresholdError {
    ParseFloat,
    OutOfRange { value: f32 },
}

#[derive(Clone, Copy, Debug)]
struct SearchPropertyConfig {
    min_recall: f32,
}

impl SearchPropertyConfig {
    const ENV_KEY: &'static str = "CHUTORO_HNSW_PBT_MIN_RECALL";
    const DEFAULT_MIN_RECALL: f32 = 0.90;

    fn load() -> Self {
        let min_recall = env::var(Self::ENV_KEY)
            .ok()
            .and_then(|value| match parse_recall_threshold(&value) {
                Ok(parsed) => Some(parsed),
                Err(err) => {
                    tracing::warn!(
                        env = Self::ENV_KEY,
                        raw = value,
                        ?err,
                        "invalid recall threshold, falling back to default",
                    );
                    None
                }
            })
            .unwrap_or(Self::DEFAULT_MIN_RECALL);
        Self { min_recall }
    }

    fn min_recall(&self) -> f32 {
        self.min_recall
    }
}

fn parse_recall_threshold(raw: &str) -> Result<f32, RecallThresholdError> {
    let trimmed = raw.trim();
    let parsed = trimmed
        .parse::<f32>()
        .map_err(|_| RecallThresholdError::ParseFloat)?;
    if !(0.0..=1.0).contains(&parsed) || parsed <= 0.0 {
        return Err(RecallThresholdError::OutOfRange { value: parsed });
    }
    Ok(parsed)
}

#[rstest]
#[case("0.5", 0.5)]
#[case("1.0", 1.0)]
#[case("0.01", 0.01)]
fn parse_recall_threshold_accepts_valid_values(#[case] input: &str, #[case] expected: f32) {
    let parsed = parse_recall_threshold(input).expect("value should parse");
    assert!(
        (parsed - expected).abs() < f32::EPSILON,
        "parsed {parsed} vs {expected}"
    );
}

#[rstest]
#[case("0.0", RecallThresholdError::OutOfRange { value: 0.0 })]
#[case("-0.1", RecallThresholdError::OutOfRange { value: -0.1 })]
#[case("1.0001", RecallThresholdError::OutOfRange { value: 1.0001 })]
#[case("abc", RecallThresholdError::ParseFloat)]
fn parse_recall_threshold_rejects_invalid_values(
    #[case] input: &str,
    #[case] expected: RecallThresholdError,
) {
    let err = parse_recall_threshold(input).expect_err("value should fail");
    assert_eq!(err, expected);
}

#[rstest]
#[case(vec![0, 1, 2], vec![0, 1, 2], 3, 1.0)]
#[case(vec![0, 1, 2], vec![0, 2, 3], 2, 0.5)]
#[case(vec![1, 2, 3], vec![4, 5, 6], 1, 0.0)]
fn recall_at_k_computes_expected_hits(
    #[case] oracle_ids: Vec<usize>,
    #[case] observed_ids: Vec<usize>,
    #[case] k: usize,
    #[case] expected: f32,
) {
    let oracle = neighbours_from_ids(&oracle_ids);
    let observed = neighbours_from_ids(&observed_ids);
    let recall = recall_at_k(&oracle, &observed, k);
    assert!(
        (recall - expected).abs() < f32::EPSILON,
        "recall {recall} vs {expected}"
    );
}

fn neighbours_from_ids(ids: &[usize]) -> Vec<Neighbour> {
    ids.iter()
        .enumerate()
        .map(|(idx, &id)| Neighbour {
            id,
            distance: idx as f32,
        })
        .collect()
}
