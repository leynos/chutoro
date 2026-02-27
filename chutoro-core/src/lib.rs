//! Chutoro core library.

mod builder;
mod chutoro;
mod clustering_quality;
#[cfg(feature = "cpu")]
mod cpu_pipeline;
mod datasource;
mod distance;
mod error;
#[cfg(feature = "cpu")]
mod hierarchy;
#[cfg(feature = "cpu")]
mod hnsw;
mod memory;
#[cfg(feature = "cpu")]
mod mst;
mod result;

pub use crate::{
    builder::{ChutoroBuilder, ExecutionStrategy},
    chutoro::Chutoro,
    clustering_quality::{
        ClusteringQualityError, ClusteringQualityScore, adjusted_rand_index,
        clustering_quality_score, normalized_mutual_information,
    },
    datasource::{DataSource, MetricDescriptor},
    distance::{
        CosineNorms, Distance, DistanceError, Norm, Result as DistanceResult, VectorKind,
        cosine_distance, euclidean_distance,
    },
    error::{ChutoroError, ChutoroErrorCode, DataSourceError, DataSourceErrorCode, Result},
    memory::{estimate_peak_bytes, format_bytes},
    result::{ClusterId, ClusteringResult, NonContiguousClusterIds},
};

#[cfg(feature = "cpu")]
pub use crate::cpu_pipeline::run_cpu_pipeline;

#[cfg(feature = "cpu")]
/// CPU-accelerated HNSW index components; requires the `cpu` feature.
pub use crate::hnsw::{
    CandidateEdge, CpuHnsw, DistanceCacheConfig, EdgeHarvest, HnswError, HnswErrorCode,
    HnswInvariant, HnswInvariantChecker, HnswInvariantViolation, HnswParams, Neighbour,
};

#[cfg(feature = "cpu")]
/// CPU minimum spanning tree (MST) utilities; requires the `cpu` feature.
pub use crate::mst::{MinimumSpanningForest, MstEdge, MstError, MstErrorCode, parallel_kruskal};

#[cfg(feature = "cpu")]
/// Hierarchy extraction utilities for the CPU pipeline; requires the `cpu` feature.
pub use crate::hierarchy::{
    HierarchyConfig, HierarchyError, HierarchyErrorCode, extract_labels_from_mst,
};

#[cfg(test)]
pub(crate) mod test_utils;
