//! Search correctness helpers for the CPU HNSW property suite.
//!
//! Hosts the brute-force oracle, recall measurement utilities, configurable
//! recall threshold parsing, and targeted unit tests that exercise the helper
//! logic directly via `rstest`.

use std::{
    collections::{BinaryHeap, HashSet},
    env,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

#[cfg(test)]
use super::types::{DistributionMetadata, HnswParamsSeed, VectorDistribution};
use proptest::{
    prop_assume,
    test_runner::{TestCaseError, TestCaseResult},
};
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

    if len > config.max_fixture_len() {
        return Ok(());
    }

    const MIN_VALID_CONNECTIONS: usize = 16;
    prop_assume!(fixture.params.max_connections >= MIN_VALID_CONNECTIONS);
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

    let threshold = config.min_recall();
    let recall = recall_at_k(&oracle, &hnsw_neighbours, k);
    let recall_ctx = RecallCheckContext {
        fixture: &fixture,
        len,
        k,
        query,
        fanout_cap,
        threshold,
    };
    record_search_metrics(SearchMetricsContext {
        fixture: &fixture,
        len,
        k,
        recall,
        threshold,
        timings: SearchTimings::new(hnsw_elapsed, oracle_elapsed),
    });

    ensure_recall_meets_threshold(recall, &recall_ctx)
}

fn brute_force_top_k<D: DataSource + Sync>(
    source: &D,
    query: usize,
    k: usize,
) -> Result<Vec<Neighbour>, DataSourceError> {
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut heap: BinaryHeap<Neighbour> = BinaryHeap::with_capacity(k);
    for candidate in 0..source.len() {
        let distance = source.distance(query, candidate)?;
        heap.push(Neighbour {
            id: candidate,
            distance,
        });
        if heap.len() > k {
            heap.pop();
        }
    }

    let mut neighbours = heap.into_vec();
    neighbours.sort_unstable();
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

struct RecallCheckContext<'a> {
    fixture: &'a HnswFixture,
    len: usize,
    k: usize,
    query: usize,
    fanout_cap: usize,
    threshold: f32,
}

fn ensure_recall_meets_threshold(recall: f32, ctx: &RecallCheckContext<'_>) -> TestCaseResult {
    if recall < ctx.threshold {
        return Err(TestCaseError::fail(format!(
            "recall {recall:.3} below threshold {threshold:.3} (len={len}, k={k}, query={query}, max_connections={}, fanout_cap={}, distribution={:?})",
            ctx.fixture.params.max_connections,
            ctx.fanout_cap,
            ctx.fixture.distribution,
            len = ctx.len,
            k = ctx.k,
            query = ctx.query,
            threshold = ctx.threshold,
        )));
    }

    Ok(())
}

fn record_search_metrics(ctx: SearchMetricsContext<'_>) {
    tracing::debug!(
        distribution = ?ctx.fixture.distribution,
        dimension = ctx.fixture.dimension(),
        len = ctx.len,
        k = ctx.k,
        recall = ctx.recall,
        threshold = ctx.threshold,
        hnsw_micros = ctx.timings.hnsw.as_micros(),
        oracle_micros = ctx.timings.oracle.as_micros(),
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
    max_fixture_len: usize,
}

impl SearchPropertyConfig {
    const ENV_KEY: &'static str = "CHUTORO_HNSW_PBT_MIN_RECALL";
    const MAX_FIXTURE_LEN_ENV_KEY: &'static str = "CHUTORO_HNSW_PBT_MAX_FIXTURE_LEN";
    const DEFAULT_MIN_RECALL: f32 = 0.50;
    const DEFAULT_MAX_FIXTURE_LEN: usize = 32;

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

        let max_fixture_len = env::var(Self::MAX_FIXTURE_LEN_ENV_KEY)
            .ok()
            .and_then(|value| {
                let trimmed = value.trim();
                match trimmed.parse::<usize>() {
                    Ok(parsed) if parsed >= 2 => Some(parsed),
                    Ok(parsed) => {
                        tracing::warn!(
                            env = Self::MAX_FIXTURE_LEN_ENV_KEY,
                            raw = %value,
                            parsed,
                            "invalid max fixture length (must be >= 2); falling back to default",
                        );
                        None
                    }
                    Err(err) => {
                        tracing::warn!(
                            env = Self::MAX_FIXTURE_LEN_ENV_KEY,
                            raw = %value,
                            ?err,
                            "invalid max fixture length; falling back to default",
                        );
                        None
                    }
                }
            })
            .unwrap_or(Self::DEFAULT_MAX_FIXTURE_LEN);

        Self {
            min_recall,
            max_fixture_len,
        }
    }

    fn min_recall(&self) -> f32 {
        self.min_recall
    }

    fn max_fixture_len(&self) -> usize {
        self.max_fixture_len
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

#[cfg(test)]
fn fixture_with_vectors(vectors: Vec<Vec<f32>>, max_connections: usize) -> HnswFixture {
    assert!(
        !vectors.is_empty(),
        "test fixtures must contain at least one vector"
    );
    let dimension = vectors[0].len();
    assert!(vectors.iter().all(|vector| vector.len() == dimension));
    HnswFixture {
        distribution: VectorDistribution::Uniform,
        vectors,
        metadata: DistributionMetadata::Uniform { bound: 1.0 },
        params: HnswParamsSeed {
            max_connections,
            ef_construction: 64,
            level_multiplier: 1.0,
            max_level: 2,
            rng_seed: 42,
        },
    }
}

#[cfg(test)]
fn uniform_fixture(max_connections: usize) -> HnswFixture {
    fixture_with_vectors(vec![vec![0.0, 0.0], vec![1.0, 1.0]], max_connections)
}

#[cfg(test)]
#[derive(Clone)]
struct MatrixSource {
    distances: Vec<Vec<f32>>,
}

#[cfg(test)]
impl MatrixSource {
    fn new(distances: Vec<Vec<f32>>) -> Self {
        Self { distances }
    }
}

#[cfg(test)]
impl DataSource for MatrixSource {
    fn len(&self) -> usize {
        self.distances.len()
    }

    fn name(&self) -> &str {
        "matrix"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let row = self
            .distances
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        row.get(j)
            .copied()
            .ok_or(DataSourceError::OutOfBounds { index: j })
    }
}

#[cfg(test)]
#[test]
fn brute_force_top_k_returns_empty_when_k_is_zero() {
    let source = MatrixSource::new(vec![vec![0.0, 0.5], vec![0.5, 0.0]]);
    let neighbours = brute_force_top_k(&source, 0, 0).expect("k=0 should succeed");
    assert!(neighbours.is_empty());
}

#[cfg(test)]
#[test]
fn brute_force_top_k_returns_all_when_k_exceeds_len() {
    let source = MatrixSource::new(vec![vec![0.0, 0.4], vec![0.4, 0.0]]);
    let neighbours = brute_force_top_k(&source, 0, 10).expect("k>len should return all nodes");
    assert_eq!(neighbours.len(), 2);
    assert_eq!(
        neighbours.iter().map(|n| n.id).collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[cfg(test)]
#[test]
fn brute_force_top_k_handles_empty_source() {
    let source = MatrixSource::new(Vec::new());
    let neighbours = brute_force_top_k(&source, 0, 1).expect("empty source should be ok");
    assert!(neighbours.is_empty());
}

#[cfg(test)]
#[test]
fn search_property_returns_ok_for_single_item_fixture() {
    let fixture = fixture_with_vectors(vec![vec![0.0]], 16);
    run_search_correctness_property(fixture, 0, 0)
        .expect("len < 2 fixture should be skipped early");
}

#[cfg(test)]
#[test]
fn recall_threshold_failure_includes_context() {
    let fixture = uniform_fixture(32);
    let ctx = RecallCheckContext {
        fixture: &fixture,
        len: 10,
        k: 4,
        query: 2,
        fanout_cap: 8,
        threshold: 0.9,
    };
    let err =
        ensure_recall_meets_threshold(0.5, &ctx).expect_err("recall below threshold must fail");

    match err {
        TestCaseError::Fail(message) => {
            let text = message.message();
            assert!(text.contains("recall 0.500 below threshold 0.900"));
            assert!(text.contains("len=10, k=4, query=2"));
            assert!(text.contains("max_connections=32"));
            assert!(text.contains("fanout_cap=8"));
            assert!(text.contains("distribution=Uniform"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
