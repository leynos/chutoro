//! Chutoro core library.
#![cfg_attr(docsrs, feature(doc_cfg))]

mod builder;
mod chutoro;
mod datasource;
mod distance;
mod error;
#[cfg(feature = "cpu")]
mod hnsw;
mod result;

pub use crate::{
    builder::{ChutoroBuilder, ExecutionStrategy},
    chutoro::Chutoro,
    datasource::DataSource,
    distance::{
        CosineNorms, Distance, DistanceError, Norm, Result as DistanceResult, VectorKind,
        cosine_distance, euclidean_distance,
    },
    error::{ChutoroError, ChutoroErrorCode, DataSourceError, DataSourceErrorCode, Result},
    result::{ClusterId, ClusteringResult, NonContiguousClusterIds},
};

#[cfg(feature = "cpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "cpu")))]
pub use crate::hnsw::{CpuHnsw, HnswError, HnswErrorCode, HnswParams, Neighbour};
