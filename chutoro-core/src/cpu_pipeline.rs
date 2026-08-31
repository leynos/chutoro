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

use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    CandidateEdge, ClusterId, CpuHnsw, DataSource, EdgeHarvest, HierarchyConfig, HnswError,
    HnswParams, MstError, Result, error::ChutoroError, parallel_kruskal, result::ClusteringResult,
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
    let configured_ef_construction = hnsw_params.ef_construction();
    let effective_hnsw_params = hnsw_params.clone().bounded_for_point_count(items);
    debug!(
        max_connections = effective_hnsw_params.max_connections(),
        configured_ef_construction,
        effective_ef_construction = effective_hnsw_params.ef_construction(),
        "building CPU HNSW index"
    );
    let (index, harvested) = CpuHnsw::build_with_edges(source, effective_hnsw_params.clone())
        .map_err(|error| map_cpu_hnsw_error(source, error))?;

    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(effective_hnsw_params.ef_construction())
        .min(items);
    let ef = NonZeroUsize::new(desired).unwrap_or(NonZeroUsize::MIN);
    let core_distance_inputs = CoreDistanceInputs::new(items, min_cluster_size, ef);
    let core_distances = compute_core_distances(source, &index, &core_distance_inputs)?;
    let mutual_edges: Vec<CandidateEdge> = harvested
        .iter()
        .map(|edge| {
            let left = edge.source();
            let right = edge.target();
            let dist = edge.distance();
            let left_core_distance = core_distances.get(left).copied().ok_or_else(|| {
                map_cpu_hnsw_error(
                    source,
                    HnswError::GraphInvariantViolation {
                        message: format!("harvested edge source {left} has no core distance"),
                    },
                )
            })?;
            let right_core_distance = core_distances.get(right).copied().ok_or_else(|| {
                map_cpu_hnsw_error(
                    source,
                    HnswError::GraphInvariantViolation {
                        message: format!("harvested edge target {right} has no core distance"),
                    },
                )
            })?;
            let weight = dist.max(left_core_distance).max(right_core_distance);
            Ok(CandidateEdge::new(left, right, weight, edge.sequence()))
        })
        .collect::<Result<_>>()?;
    let mutual_harvest = EdgeHarvest::new(mutual_edges);

    let forest =
        parallel_kruskal(items, &mutual_harvest).map_err(|error| map_cpu_mst_error(&error))?;
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
            .map_err(|error| map_cpu_hnsw_error(source, error))?;
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

/// Translate an HNSW failure into the public CPU-pipeline error type.
#[cfg(feature = "cpu")]
pub(crate) fn map_cpu_hnsw_error<D: DataSource>(source: &D, hnsw_error: HnswError) -> ChutoroError {
    match hnsw_error {
        HnswError::DataSource(data_source_error) => ChutoroError::DataSource {
            data_source: Arc::from(source.name()),
            error: data_source_error,
        },
        other_error => ChutoroError::CpuHnswFailure {
            code: Arc::from(other_error.code().as_str()),
            message: Arc::from(other_error.to_string()),
        },
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
