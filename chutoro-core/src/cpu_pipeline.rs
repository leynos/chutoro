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

/// Runs the CPU pipeline end-to-end for the provided [`DataSource`].
///
/// # Errors
/// Returns the same errors as [`crate::Chutoro::run`], including empty or
/// undersized sources, data source failures, and CPU pipeline failures.
#[cfg(feature = "cpu")]
pub fn run_cpu_pipeline<D: DataSource + Sync>(
    source: &D,
    min_cluster_size: NonZeroUsize,
) -> Result<ClusteringResult> {
    let items = source.len();
    if items == 0 {
        return Err(ChutoroError::EmptySource {
            data_source: Arc::from(source.name()),
        });
    }
    if items < min_cluster_size.get() {
        return Err(ChutoroError::InsufficientItems {
            data_source: Arc::from(source.name()),
            items,
            min_cluster_size,
        });
    }

    run_cpu_pipeline_with_len(source, items, min_cluster_size)
}

/// Run the CPU pipeline after source-length validation.
#[cfg(feature = "cpu")]
pub(crate) fn run_cpu_pipeline_with_len<D: DataSource + Sync>(
    source: &D,
    items: usize,
    min_cluster_size: NonZeroUsize,
) -> Result<ClusteringResult> {
    let params = HnswParams::default();
    let (index, harvested) = CpuHnsw::build_with_edges(source, params.clone())
        .map_err(|error| map_cpu_hnsw_error(source, error))?;

    let desired = min_cluster_size
        .get()
        .saturating_add(1)
        .max(params.ef_construction())
        .min(items);
    let ef = NonZeroUsize::new(desired).unwrap_or(NonZeroUsize::MIN);

    let mut core_distances = Vec::with_capacity(items);
    for point in 0..items {
        let neighbours = index
            .search(source, point, ef)
            .map_err(|error| map_cpu_hnsw_error(source, error))?;
        let others: Vec<_> = neighbours.into_iter().filter(|n| n.id != point).collect();
        let core = others
            .get(min_cluster_size.get().saturating_sub(1))
            .or_else(|| others.last())
            .map_or(0.0, |neighbour| neighbour.distance);
        core_distances.push(core);
    }

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
