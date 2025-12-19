//! Chutoro core library.

mod builder;
mod chutoro;
mod datasource;
mod distance;
mod error;
#[cfg(feature = "cpu")]
mod hierarchy;
#[cfg(feature = "cpu")]
mod hnsw;
#[cfg(feature = "cpu")]
mod mst;
mod result;

pub use crate::{
    builder::{ChutoroBuilder, ExecutionStrategy},
    chutoro::Chutoro,
    datasource::{DataSource, MetricDescriptor},
    distance::{
        CosineNorms, Distance, DistanceError, Norm, Result as DistanceResult, VectorKind,
        cosine_distance, euclidean_distance,
    },
    error::{ChutoroError, ChutoroErrorCode, DataSourceError, DataSourceErrorCode, Result},
    result::{ClusterId, ClusteringResult, NonContiguousClusterIds},
};

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
