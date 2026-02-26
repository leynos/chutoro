//! Clustering-quality pass helpers for the HNSW `ef_construction` sweep.

use std::{num::NonZeroUsize, path::PathBuf, time::Instant};

use chutoro_benches::{
    clustering_quality::{
        ClusteringQualityMeasurement, clustering_quality_score, write_clustering_quality_report,
    },
    ef_sweep::{
        BENCH_SEED, EF_CONSTRUCTION_VALUES, EF_SWEEP_MAX_CONNECTIONS, make_hnsw_params_with_ef,
        resolve_ef_construction,
    },
    error::BenchSetupError,
    source::{Anisotropy, GaussianBlobConfig, SyntheticSource},
};
use chutoro_core::{
    CandidateEdge, CpuHnsw, DataSource, EdgeHarvest, HierarchyConfig, HnswError, HnswParams,
    extract_labels_from_mst, parallel_kruskal,
};

use crate::{
    CLUSTERING_QUALITY_CLUSTER_COUNT, CLUSTERING_QUALITY_MIN_CLUSTER_SIZE,
    CLUSTERING_QUALITY_POINT_COUNT, cluster_quality_report_path,
    should_collect_cluster_quality_report,
};

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

fn compute_core_distances(
    index: &CpuHnsw,
    source: &SyntheticSource,
    ef: NonZeroUsize,
    min_cluster_size: NonZeroUsize,
) -> Result<Vec<f64>, BenchSetupError> {
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
                .map(|neighbour| f64::from(neighbour.distance))
                .ok_or_else(|| {
                    BenchSetupError::Hnsw(HnswError::GraphInvariantViolation {
                        message: format!(
                            "core distance neighbour missing at core_rank {core_rank} for \
                             min_cluster_size {} (others length {})",
                            min_cluster_size.get(),
                            others.len()
                        ),
                    })
                })?
        } else {
            if others.is_empty() {
                use std::io::Write as _;

                let mut diagnostic_stream = std::io::stderr().lock();
                let _diagnostic_write = writeln!(
                    diagnostic_stream,
                    "[diagnostic] compute_core_distances: point {point} has no non-self \
                     neighbours (source.len() = {})",
                    source.len()
                );
            }
            others
                .last()
                .map_or(0.0, |neighbour| f64::from(neighbour.distance))
        };
        core_distances.push(core_distance);
    }
    Ok(core_distances)
}

fn build_mutual_edges(
    harvested: &EdgeHarvest,
    core_distances: &[f64],
) -> Result<Vec<CandidateEdge>, BenchSetupError> {
    let mut mutual_edges = Vec::<CandidateEdge>::with_capacity(harvested.len());
    for edge in harvested.iter() {
        let left = edge.source();
        let right = edge.target();
        let left_core = *core_distances.get(left).ok_or_else(|| {
            BenchSetupError::Hnsw(HnswError::GraphInvariantViolation {
                message: format!(
                    "CpuHnsw::build_with_edges produced left edge index {left} out of \
                     bounds for core_distances length {}",
                    core_distances.len()
                ),
            })
        })?;
        let right_core = *core_distances.get(right).ok_or_else(|| {
            BenchSetupError::Hnsw(HnswError::GraphInvariantViolation {
                message: format!(
                    "CpuHnsw::build_with_edges produced right edge index {right} out of \
                     bounds for core_distances length {}",
                    core_distances.len()
                ),
            })
        })?;
        let mutual_distance_f64 = f64::from(edge.distance()).max(left_core).max(right_core);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "core_distances values originate from f32 neighbour distances"
        )]
        let mutual_distance_f32 = mutual_distance_f64 as f32;
        mutual_edges.push(CandidateEdge::new(
            left,
            right,
            mutual_distance_f32,
            edge.sequence(),
        ));
    }
    Ok(mutual_edges)
}

/// Builds HNSW labels for one parameter set by running:
/// `CpuHnsw::build_with_edges` -> `compute_core_distances` ->
/// `build_mutual_edges` -> `parallel_kruskal` -> `extract_labels_from_mst`.
///
/// Returns `(labels, build_time_millis)`, where `labels` contains one cluster
/// label per source item and `build_time_millis` is the HNSW build duration in
/// milliseconds.
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

    let core_distances = compute_core_distances(&index, source, ef, min_cluster_size)?;
    let mutual_edges = build_mutual_edges(&harvested, &core_distances)?;
    let forest = parallel_kruskal(source.len(), &EdgeHarvest::new(mutual_edges))?;
    let labels = extract_labels_from_mst(
        source.len(),
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )?;

    Ok((labels, build_time_millis))
}

/// Runs one optional cluster-quality sweep across configured HNSW `(M, ef)` pairs.
///
/// Returns `Ok(Some(PathBuf))` when reporting is enabled and the CSV write
/// succeeds, `Ok(None)` when reporting is disabled, and `Err(BenchSetupError)`
/// for setup, pipeline, scoring, or report-write failures.
///
/// Side effects:
/// - Generates synthetic labelled data.
/// - Builds HNSW indices and computes ARI/NMI measurements.
/// - Writes a clustering-quality report file when enabled.
///
/// Preconditions and guarantees:
/// - Requires non-zero `CLUSTERING_QUALITY_MIN_CLUSTER_SIZE`.
/// - Uses deterministic benchmark fixtures/parameters for reproducible output.
pub(super) fn measure_clustering_quality_vs_ef_impl() -> Result<Option<PathBuf>, BenchSetupError> {
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
