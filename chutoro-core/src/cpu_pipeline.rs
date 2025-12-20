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
    let ef = NonZeroUsize::new(desired)
        .expect("ef_construction is non-zero so the computed ef is non-zero");

    let mut core_distances = Vec::with_capacity(items);
    for point in 0..items {
        let neighbours = index
            .search(source, point, ef)
            .map_err(|error| map_cpu_hnsw_error(source, error))?;
        let others: Vec<_> = neighbours.into_iter().filter(|n| n.id != point).collect();
        let core = if others.len() >= min_cluster_size.get() {
            others[min_cluster_size.get() - 1].distance
        } else {
            others.last().map(|n| n.distance).unwrap_or(0.0)
        };
        core_distances.push(core);
    }

    let mutual_edges: Vec<CandidateEdge> = harvested
        .iter()
        .map(|edge| {
            let left = edge.source();
            let right = edge.target();
            let dist = edge.distance();
            let weight = dist.max(core_distances[left]).max(core_distances[right]);
            CandidateEdge::new(left, right, weight, edge.sequence())
        })
        .collect();
    let mutual_harvest = EdgeHarvest::new(mutual_edges);

    let forest = parallel_kruskal(items, &mutual_harvest).map_err(map_cpu_mst_error)?;

    let labels = crate::extract_labels_from_mst(
        items,
        forest.edges(),
        HierarchyConfig::new(min_cluster_size),
    )
    .map_err(map_cpu_hierarchy_error)?;

    let assignments = labels
        .into_iter()
        .map(|label| ClusterId::new(label as u64))
        .collect();

    Ok(ClusteringResult::from_assignments(assignments))
}

#[cfg(feature = "cpu")]
fn map_cpu_hnsw_error<D: DataSource>(source: &D, error: HnswError) -> ChutoroError {
    match error {
        HnswError::DataSource(error) => ChutoroError::DataSource {
            data_source: Arc::from(source.name()),
            error,
        },
        other => ChutoroError::CpuHnswFailure {
            code: Arc::from(other.code().as_str()),
            message: Arc::from(other.to_string()),
        },
    }
}

#[cfg(feature = "cpu")]
fn map_cpu_mst_error(error: MstError) -> ChutoroError {
    ChutoroError::CpuMstFailure {
        code: Arc::from(error.code().as_str()),
        message: Arc::from(error.to_string()),
    }
}

#[cfg(feature = "cpu")]
fn map_cpu_hierarchy_error(error: crate::HierarchyError) -> ChutoroError {
    ChutoroError::CpuHierarchyFailure {
        code: Arc::from(error.code().as_str()),
        message: Arc::from(error.to_string()),
    }
}
