//! Chutoro core library.
#![cfg_attr(docsrs, feature(doc_cfg))]

mod builder;
mod chutoro;
mod datasource;
mod distance;
mod error;
mod result;

pub use crate::{
    builder::{ChutoroBuilder, ExecutionStrategy},
    chutoro::Chutoro,
    datasource::DataSource,
    distance::{
        CosineNorms, DistanceError, Result as DistanceResult, VectorKind, cosine_distance,
        euclidean_distance,
    },
    error::{ChutoroError, DataSourceError, Result},
    result::{ClusterId, ClusteringResult, NonContiguousClusterIds},
};
