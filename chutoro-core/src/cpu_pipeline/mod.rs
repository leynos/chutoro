//! CPU FISHDBC pipeline orchestration helpers.
//!
//! This module exists to centralize the core CPU pipeline steps so they can be
//! reused across `Chutoro` orchestration and tests:
//!
//! - Build an HNSW index while harvesting candidate edges.
//! - Convert harvested edges to mutual-reachability weights using core
//!   distances computed from HNSW neighbourhoods.
//! - Build the mutual-reachability minimum spanning forest (Kruskal).
//! - Extract a flat clustering from the mutual-reachability MST.

use std::{
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};

use rayon::prelude::*;

use crate::{
    CandidateEdge, ClusterId, CpuHnsw, DataSource, EdgeHarvest, HierarchyConfig, HnswError,
    HnswParams, MinimumSpanningForest, MstError, Result, error::ChutoroError,
    mst::parallel_kruskal_from_edges, result::ClusteringResult,
};
use tracing::debug;

/// Run the CPU pipeline after source-length validation.
#[cfg(feature = "cpu")]
pub(crate) fn run_cpu_pipeline_with_len<D: DataSource + Sync>(
    source: &D,
    items: usize,
    min_cluster_size: NonZeroUsize,
    hnsw_params: &HnswParams,
) -> Result<ClusteringResult> {
    let (harvested, core_distances) =
        build_index_and_core_distances(source, items, min_cluster_size, hnsw_params)?;
    let forest =
        build_mutual_reachability_forest(source.name(), items, &harvested, &core_distances)?;

    let labels = crate::extract_labels_from_mst(
        items,
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )
    .map_err(|error| map_cpu_hierarchy_error(&error))?;

    let assignments = labels
        .into_iter()
        .map(|label| ClusterId::new(label as u64))
        .collect();

    Ok(ClusteringResult::from_assignments(assignments))
}

/// Builds the HNSW index and the core distances that later stages consume.
///
/// The index is dropped once every core distance has been read, because no
/// later stage needs the graph itself.
#[cfg(feature = "cpu")]
fn build_index_and_core_distances<D: DataSource + Sync>(
    source: &D,
    items: usize,
    min_cluster_size: NonZeroUsize,
    hnsw_params: &HnswParams,
) -> Result<(EdgeHarvest, Vec<f32>)> {
    let configured_ef_construction = hnsw_params.ef_construction();
    let effective_hnsw_params = hnsw_params.clone().bounded_for_point_count(items);
    debug!(
        max_connections = effective_hnsw_params.max_connections(),
        configured_ef_construction,
        effective_ef_construction = effective_hnsw_params.ef_construction(),
        "building CPU HNSW index"
    );

    let (index, harvested) = CpuHnsw::build_with_edges(source, effective_hnsw_params.clone())
        .map_err(|error| error.into_chutoro_error(Arc::from(source.name())))?;

    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(effective_hnsw_params.ef_construction())
        .min(items);
    let ef = NonZeroUsize::new(desired).unwrap_or(NonZeroUsize::MIN);
    let core_distance_inputs = CoreDistanceInputs::new(items, min_cluster_size, ef);
    let core_distances = compute_core_distances(source, &index, &core_distance_inputs)?;

    Ok((harvested, core_distances))
}

/// Builds the mutual-reachability minimum spanning forest.
///
/// The weighting transform and the Kruskal stage are timed separately so the
/// batch-latency histogram can attribute time to each stage rather than to the
/// pipeline as a whole.
#[cfg(feature = "cpu")]
fn build_mutual_reachability_forest(
    source_name: &str,
    items: usize,
    harvested: &EdgeHarvest,
    core_distances: &[f32],
) -> Result<MinimumSpanningForest> {
    let transform_started = Instant::now();
    let mutual_edges = mutual_reachability_edges(source_name, harvested, core_distances)?;
    record_stage(
        PipelineStage::MutualReachability.as_label(),
        transform_started.elapsed(),
    );

    let kruskal_started = Instant::now();
    let forest = parallel_kruskal_from_edges(items, &mutual_edges)
        .map_err(|error| map_cpu_mst_error(&error))?;
    record_stage(PipelineStage::Kruskal.as_label(), kruskal_started.elapsed());

    Ok(forest)
}

/// Applies mutual-reachability weighting to every harvested candidate edge.
///
/// An edge keeps its endpoints and insertion sequence; its weight becomes the
/// maximum of the raw distance and both endpoints' core distances. The
/// transform preserves input order, so the parallel slice walk and a
/// sequential walk over [`EdgeHarvest::iter`] return the same vector.
#[cfg(feature = "cpu")]
fn mutual_reachability_edges(
    source_name: &str,
    harvested: &EdgeHarvest,
    core_distances: &[f32],
) -> Result<Vec<CandidateEdge>> {
    harvested
        .as_slice()
        .par_iter()
        .map(|edge| mutual_reachability_edge(source_name, edge, core_distances))
        .collect()
}

/// Applies mutual-reachability weighting to one candidate edge.
#[cfg(feature = "cpu")]
fn mutual_reachability_edge(
    source_name: &str,
    edge: &CandidateEdge,
    core_distances: &[f32],
) -> Result<CandidateEdge> {
    let left = edge.source();
    let right = edge.target();
    let left_core_distance = core_distances
        .get(left)
        .copied()
        .ok_or_else(|| missing_core_distance(source_name, "source", left))?;
    let right_core_distance = core_distances
        .get(right)
        .copied()
        .ok_or_else(|| missing_core_distance(source_name, "target", right))?;
    let weight = edge
        .distance()
        .max(left_core_distance)
        .max(right_core_distance);

    Ok(CandidateEdge::new(left, right, weight, edge.sequence()))
}

/// Builds the error reported when a harvested endpoint has no core distance.
#[cfg(feature = "cpu")]
fn missing_core_distance(source_name: &str, endpoint: &str, node: usize) -> ChutoroError {
    HnswError::GraphInvariantViolation {
        message: format!("harvested edge {endpoint} {node} has no core distance"),
    }
    .into_chutoro_error(Arc::from(source_name))
}

/// Pipeline stages that report a wall-clock latency.
///
/// The variants are the complete label vocabulary for the
/// `chutoro.cpu_pipeline.stage_seconds` histogram, which keeps that metric's
/// cardinality bounded.
#[cfg(feature = "cpu")]
#[derive(Clone, Copy)]
enum PipelineStage {
    /// Mutual-reachability weighting of the harvested candidate edges.
    MutualReachability,
    /// Minimum spanning forest construction over the weighted edges.
    Kruskal,
}

#[cfg(feature = "cpu")]
impl PipelineStage {
    /// Returns the stable metric label for this stage.
    const fn as_label(self) -> &'static str {
        match self {
            Self::MutualReachability => "mutual_reachability",
            Self::Kruskal => "kruskal",
        }
    }
}

/// Records a stage duration when the `metrics` feature is compiled in.
#[cfg(all(feature = "cpu", feature = "metrics"))]
fn record_stage(label: &'static str, elapsed: Duration) {
    crate::batch_metrics::record_pipeline_stage(label, elapsed);
}

/// Discards a stage timing when the `metrics` feature is not compiled in.
#[cfg(all(feature = "cpu", not(feature = "metrics")))]
const fn record_stage(_label: &'static str, _elapsed: Duration) {}

/// Computes every point's core distance from its nearest non-self neighbours.
#[cfg(feature = "cpu")]
fn compute_core_distances<D: DataSource + Sync>(
    source: &D,
    index: &CpuHnsw,
    inputs: &CoreDistanceInputs,
) -> Result<Vec<f32>> {
    let mut core_distances = Vec::with_capacity(inputs.items);
    for point in 0..inputs.items {
        let neighbours = index
            .search(source, point, inputs.ef)
            .map_err(|error| error.into_chutoro_error(Arc::from(source.name())))?;
        let others: Vec<_> = neighbours.into_iter().filter(|n| n.id != point).collect();
        let core = others
            .get(inputs.min_cluster_size.get().saturating_sub(1))
            .or_else(|| others.last())
            .map_or(0.0, |neighbour| neighbour.distance);
        core_distances.push(core);
    }

    Ok(core_distances)
}

/// Groups the controls for a single CPU batch core-distance computation.
///
/// This private type serves only [`compute_core_distances`]; do not reuse it
/// outside this pipeline boundary.
#[cfg(feature = "cpu")]
struct CoreDistanceInputs {
    /// Number of source points to process.
    items: usize,
    /// Requested non-self neighbour rank.
    min_cluster_size: NonZeroUsize,
    /// Search width used for every point lookup.
    ef: NonZeroUsize,
}

#[cfg(feature = "cpu")]
impl CoreDistanceInputs {
    /// Creates inputs for one batch core-distance computation.
    const fn new(items: usize, min_cluster_size: NonZeroUsize, ef: NonZeroUsize) -> Self {
        Self {
            items,
            min_cluster_size,
            ef,
        }
    }
}

/// Translate an MST failure into the public CPU-pipeline error type.
#[cfg(feature = "cpu")]
fn map_cpu_mst_error(error: &MstError) -> ChutoroError {
    ChutoroError::CpuMstFailure {
        code: Arc::from(error.code().as_str()),
        message: Arc::from(error.to_string()),
    }
}

/// Translate a hierarchy failure into the public CPU-pipeline error type.
#[cfg(feature = "cpu")]
fn map_cpu_hierarchy_error(error: &crate::HierarchyError) -> ChutoroError {
    ChutoroError::CpuHierarchyFailure {
        code: Arc::from(error.code().as_str()),
        message: Arc::from(error.to_string()),
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests;
