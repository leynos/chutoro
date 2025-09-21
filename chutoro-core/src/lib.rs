//! Chutoro core library.

mod builder;
mod chutoro;
mod datasource;
mod error;
mod result;

pub use crate::{
    builder::{ChutoroBuilder, ExecutionStrategy},
    chutoro::Chutoro,
    datasource::DataSource,
    error::{ChutoroError, DataSourceError},
    result::{ClusterId, ClusteringResult, NonContiguousClusterIds},
};
