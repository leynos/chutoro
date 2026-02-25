//! HNSW `ef_construction` sweep benchmarks and recall measurement.
//!
//! Varies `ef_construction` independently of `M` to show build-time versus
//! recall trade-offs. Complements the main HNSW benchmarks in `hnsw.rs`
//! which hold `ef_construction = M * 2` fixed.
use std::{num::NonZeroUsize, path::PathBuf, time::Instant};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_main};

use chutoro_benches::{
    clustering_quality::{
        ClusteringQualityMeasurement, clustering_quality_score, write_clustering_quality_report,
    },
    ef_sweep::{
        BENCH_SEED, EF_CONSTRUCTION_VALUES, EF_SWEEP_MAX_CONNECTIONS, EF_SWEEP_POINT_COUNTS,
        make_bench_source, make_hnsw_params_with_ef, resolve_ef_construction,
    },
    error::BenchSetupError,
    params::HnswBenchParams,
    recall::{RecallMeasurement, RecallScore, brute_force_top_k, recall_at_k, write_recall_report},
    source::{Anisotropy, GaussianBlobConfig, SyntheticSource},
};
use chutoro_core::{
    CandidateEdge, CpuHnsw, DataSource, EdgeHarvest, HierarchyConfig, HnswParams,
    extract_labels_from_mst, parallel_kruskal,
};

/// Dataset size for recall measurement.
const RECALL_POINT_COUNT: usize = 1_000;

/// Number of deterministic query points for recall measurement.
const RECALL_QUERY_COUNT: usize = 50;

/// Number of nearest neighbours compared for recall@k.
const RECALL_K: usize = 10;

/// Search beam width when querying the HNSW index for recall.
const RECALL_EF_SEARCH: usize = 64;

/// Report destination for recall-versus-ef_construction metrics.
const RECALL_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_recall_vs_ef.csv"
);

/// Dataset size for clustering-quality measurement.
const CLUSTERING_QUALITY_POINT_COUNT: usize = 1_000;

/// Number of Gaussian centroids used for clustering-quality measurement.
const CLUSTERING_QUALITY_CLUSTER_COUNT: usize = 8;

/// Minimum cluster size used for hierarchy extraction in quality measurement.
const CLUSTERING_QUALITY_MIN_CLUSTER_SIZE: usize = 5;

/// Report destination for ARI/NMI-versus-ef_construction metrics.
const CLUSTERING_QUALITY_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_cluster_quality_vs_ef.csv"
);

/// Panics on HNSW build failure within a Criterion benchmark closure.
///
/// Exists as a separate function (rather than inline `panic!`) because the
/// enclosing benchmark function returns `Result`, triggering Clippy's
/// `panic_in_result_fn` lint if a `panic!` appears directly in its body.
fn unwrap_build(result: Result<CpuHnsw, chutoro_core::HnswError>, context: &str) {
    if let Err(err) = result {
        panic!("{context}: {err}");
    }
}

// -- Recall measurement ------------------------------------------------

/// Emits a warning to stderr for an unrecognised env var value.
#[expect(
    clippy::print_stderr,
    reason = "Benchmark-only diagnostic for invalid env var; no structured logging available."
)]
fn warn_unrecognised_recall_env(value: &str) {
    eprintln!(
        "warning: unrecognised value {value:?} for \
         CHUTORO_BENCH_HNSW_RECALL_REPORT; expected 0/1/true/false/on/off"
    );
}

fn should_collect_recall_report() -> bool {
    if let Ok(value) = std::env::var("CHUTORO_BENCH_HNSW_RECALL_REPORT") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "0" | "false" | "off") {
            return false;
        }
        if matches!(normalized.as_str(), "1" | "true" | "on") {
            return true;
        }
        warn_unrecognised_recall_env(&value);
    }
    !std::env::args().any(|arg| arg == "--list" || arg == "--exact")
}

fn recall_report_path() -> PathBuf {
    std::env::var_os("CHUTORO_BENCH_HNSW_RECALL_REPORT_PATH")
        .map_or_else(|| PathBuf::from(RECALL_REPORT_PATH), PathBuf::from)
}

#[expect(
    clippy::print_stderr,
    reason = "Benchmark-only diagnostic for invalid env var; no structured logging available."
)]
fn warn_unrecognised_cluster_quality_env(value: &str) {
    eprintln!(
        "warning: unrecognised value {value:?} for \
         CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT; expected 0/1/true/false/on/off"
    );
}

fn should_collect_cluster_quality_report() -> bool {
    if let Ok(value) = std::env::var("CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "0" | "false" | "off") {
            return false;
        }
        if matches!(normalized.as_str(), "1" | "true" | "on") {
            return true;
        }
        warn_unrecognised_cluster_quality_env(&value);
    }
    !std::env::args().any(|arg| arg == "--list" || arg == "--exact")
}

fn cluster_quality_report_path() -> PathBuf {
    std::env::var_os("CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH").map_or_else(
        || PathBuf::from(CLUSTERING_QUALITY_REPORT_PATH),
        PathBuf::from,
    )
}

/// Returns an evenly-spaced query index for deterministic recall sampling.
///
/// Maps query ordinal `qi` in `0..RECALL_QUERY_COUNT` to an index in
/// `0..len` using `(qi + 1) * len / (count + 1)`, which avoids bias
/// towards lower indices and never returns index 0 or `len - 1` at the
/// extremes (unless the dataset is very small).
#[expect(
    clippy::integer_division,
    clippy::integer_division_remainder_used,
    reason = "Intentional truncating division to produce evenly-spaced indices."
)]
const fn query_index(qi: usize, len: usize) -> usize {
    qi.saturating_add(1).saturating_mul(len) / RECALL_QUERY_COUNT.saturating_add(1)
}

fn collect_recall_over_queries(
    source: &SyntheticSource,
    index: &CpuHnsw,
    ef_search: NonZeroUsize,
) -> Result<RecallScore, BenchSetupError> {
    let len = source.len();
    let mut total_hits: usize = 0;
    let mut total_targets: usize = 0;

    for qi in 0..RECALL_QUERY_COUNT {
        let query = query_index(qi, len);
        let mut hnsw_result = index.search(source, query, ef_search)?;
        hnsw_result.retain(|n| n.id != query);
        hnsw_result.truncate(RECALL_K);
        let oracle = brute_force_top_k(source, query, RECALL_K)?;
        let score = recall_at_k(&oracle, &hnsw_result, RECALL_K);
        total_hits = total_hits.saturating_add(score.hits);
        total_targets = total_targets.saturating_add(score.total);
    }

    Ok(RecallScore {
        hits: total_hits,
        total: total_targets,
    })
}

fn measure_recall_vs_ef_impl() -> Result<Option<PathBuf>, BenchSetupError> {
    if !should_collect_recall_report() {
        return Ok(None);
    }

    let source = make_bench_source(RECALL_POINT_COUNT)?;
    let ef_search = NonZeroUsize::new(RECALL_EF_SEARCH).ok_or(BenchSetupError::ZeroValue {
        context: "RECALL_EF_SEARCH",
    })?;
    let mut records = Vec::new();

    for &m in EF_SWEEP_MAX_CONNECTIONS {
        for &ef_raw in EF_CONSTRUCTION_VALUES {
            let ef = resolve_ef_construction(m, ef_raw);
            let params = make_hnsw_params_with_ef(m, ef, BENCH_SEED)?;

            let started = Instant::now();
            let index = CpuHnsw::build(&source, params)?;
            let build_time = started.elapsed();

            let recall = collect_recall_over_queries(&source, &index, ef_search)?;

            records.push(RecallMeasurement {
                point_count: RECALL_POINT_COUNT,
                max_connections: m,
                ef_construction: ef,
                recall,
                build_time_millis: build_time.as_millis(),
            });
        }
    }

    write_recall_report(recall_report_path(), &records)
        .map(Some)
        .map_err(BenchSetupError::RecallReport)
}

// -- Clustering-quality measurement ------------------------------------

fn make_cluster_quality_source() -> Result<(SyntheticSource, Vec<usize>), BenchSetupError> {
    SyntheticSource::generate_gaussian_blobs_with_labels(&GaussianBlobConfig {
        point_count: CLUSTERING_QUALITY_POINT_COUNT,
        dimensions: chutoro_benches::ef_sweep::BENCH_DIMENSIONS,
        cluster_count: CLUSTERING_QUALITY_CLUSTER_COUNT,
        separation: 6.0,
        anisotropy: Anisotropy::Isotropic(0.35),
        seed: BENCH_SEED,
    })
    .map_err(BenchSetupError::from)
}

fn pipeline_labels_with_hnsw_params(
    source: &SyntheticSource,
    params: &HnswParams,
    min_cluster_size: NonZeroUsize,
) -> Result<(Vec<usize>, u128), BenchSetupError> {
    let started = Instant::now();
    let (index, harvested) = CpuHnsw::build_with_edges(source, params.clone())?;
    let build_time_millis = started.elapsed().as_millis();

    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(params.ef_construction())
        .min(source.len());
    let ef = NonZeroUsize::new(desired).ok_or(BenchSetupError::ZeroValue {
        context: "cluster-quality ef",
    })?;

    let mut core_distances = Vec::with_capacity(source.len());
    for point in 0..source.len() {
        let neighbours = index.search(source, point, ef)?;
        let others: Vec<_> = neighbours
            .into_iter()
            .filter(|neighbour| neighbour.id != point)
            .collect();
        let core_distance = if others.len() >= min_cluster_size.get() {
            let core_rank = min_cluster_size.get().saturating_sub(1);
            others
                .get(core_rank)
                .map_or(0.0, |neighbour| neighbour.distance)
        } else {
            others.last().map_or(0.0, |neighbour| neighbour.distance)
        };
        core_distances.push(core_distance);
    }

    let mut mutual_edges = Vec::<CandidateEdge>::with_capacity(harvested.len());
    for edge in harvested.iter() {
        let left = edge.source();
        let right = edge.target();
        let Some(&left_core) = core_distances.get(left) else {
            continue;
        };
        let Some(&right_core) = core_distances.get(right) else {
            continue;
        };
        let mutual_distance = edge.distance().max(left_core).max(right_core);
        mutual_edges.push(CandidateEdge::new(
            left,
            right,
            mutual_distance,
            edge.sequence(),
        ));
    }
    let forest = parallel_kruskal(source.len(), &EdgeHarvest::new(mutual_edges))?;
    let labels = extract_labels_from_mst(
        source.len(),
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )?;

    Ok((labels, build_time_millis))
}

fn measure_clustering_quality_vs_ef_impl() -> Result<Option<PathBuf>, BenchSetupError> {
    if !should_collect_cluster_quality_report() {
        return Ok(None);
    }

    let (source, ground_truth_labels) = make_cluster_quality_source()?;
    let min_cluster_size = NonZeroUsize::new(CLUSTERING_QUALITY_MIN_CLUSTER_SIZE).ok_or(
        BenchSetupError::ZeroValue {
            context: "CLUSTERING_QUALITY_MIN_CLUSTER_SIZE",
        },
    )?;
    let mut records = Vec::new();

    for &m in EF_SWEEP_MAX_CONNECTIONS {
        for &ef_raw in EF_CONSTRUCTION_VALUES {
            let ef = resolve_ef_construction(m, ef_raw);
            let params = make_hnsw_params_with_ef(m, ef, BENCH_SEED)?;
            let (predicted_labels, build_time_millis) =
                pipeline_labels_with_hnsw_params(&source, &params, min_cluster_size)?;
            let score = clustering_quality_score(&ground_truth_labels, &predicted_labels)?;

            records.push(ClusteringQualityMeasurement {
                point_count: CLUSTERING_QUALITY_POINT_COUNT,
                max_connections: m,
                ef_construction: ef,
                min_cluster_size: min_cluster_size.get(),
                ari: score.ari,
                nmi: score.nmi,
                build_time_millis,
            });
        }
    }

    write_clustering_quality_report(cluster_quality_report_path(), &records)
        .map(Some)
        .map_err(BenchSetupError::ClusteringQualityReport)
}

// -- Criterion ef_construction sweep -----------------------------------

#[expect(
    clippy::excessive_nesting,
    reason = "Criterion bench_with_input + triple parameter loop requires deep nesting"
)]
fn hnsw_build_ef_sweep_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let _maybe_recall_report_path = measure_recall_vs_ef_impl()?;
    let _maybe_quality_report_path = measure_clustering_quality_vs_ef_impl()?;

    let mut group = c.benchmark_group("hnsw_build_ef_sweep");
    group.sample_size(10);

    for &point_count in EF_SWEEP_POINT_COUNTS {
        let source = make_bench_source(point_count)?;
        for &m in EF_SWEEP_MAX_CONNECTIONS {
            for &ef_raw in EF_CONSTRUCTION_VALUES {
                let ef = resolve_ef_construction(m, ef_raw);
                let bench_params = HnswBenchParams {
                    point_count,
                    max_connections: m,
                    ef_construction: ef,
                };
                let params = make_hnsw_params_with_ef(m, ef, BENCH_SEED)?;
                group.bench_with_input(
                    BenchmarkId::from_parameter(&bench_params),
                    &params,
                    |b, input_params| {
                        b.iter_batched(
                            || input_params.clone(),
                            |cloned_params| {
                                unwrap_build(
                                    CpuHnsw::build(&source, cloned_params),
                                    "hnsw_build_ef_sweep failed during benchmark",
                                );
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }

    group.finish();
    Ok(())
}

fn hnsw_build_ef_sweep(c: &mut Criterion) {
    if let Err(err) = hnsw_build_ef_sweep_impl(c) {
        panic!("hnsw_build_ef_sweep benchmark setup failed: {err}");
    }
}

mod bench_harness {
    use super::hnsw_build_ef_sweep;
    use criterion::criterion_group;

    criterion_group!(benches, hnsw_build_ef_sweep);
}
criterion_main!(bench_harness::benches);
