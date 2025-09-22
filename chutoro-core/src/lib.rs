//! Chutoro core library.
#![cfg_attr(docsrs, feature(doc_cfg))]

mod builder;
mod chutoro;
mod datasource;
mod error;
mod result;

pub use crate::{
    builder::{ChutoroBuilder, ExecutionStrategy},
    chutoro::Chutoro,
    datasource::DataSource,
    error::{ChutoroError, DataSourceError, Result},
    result::{ClusterId, ClusteringResult, NonContiguousClusterIds},
};
