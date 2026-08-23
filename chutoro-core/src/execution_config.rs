//! Validated execution policy shared by batch and session construction.
//!
//! `ExecutionConfig` is an internal composition value. The builder creates it
//! once after validation, then passes it to either `Chutoro` or `SessionConfig`
//! so both execution paths preserve the same clustering policy.

use std::num::NonZeroUsize;

#[cfg(feature = "cpu")]
use crate::HnswParams;

/// Validated clustering settings shared by all execution paths.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExecutionConfig {
    min_cluster_size: NonZeroUsize,
    #[cfg(feature = "cpu")]
    hnsw_params: HnswParams,
}

impl ExecutionConfig {
    /// Creates a configuration from values already validated by the builder.
    #[cfg(feature = "cpu")]
    pub(crate) fn new(min_cluster_size: NonZeroUsize, hnsw_params: HnswParams) -> Self {
        Self {
            min_cluster_size,
            hnsw_params,
        }
    }

    /// Creates a configuration from values already validated by the builder.
    #[cfg(not(feature = "cpu"))]
    pub(crate) fn new(min_cluster_size: NonZeroUsize) -> Self {
        Self { min_cluster_size }
    }

    /// Returns the validated minimum cluster size.
    pub(crate) const fn min_cluster_size(&self) -> NonZeroUsize {
        self.min_cluster_size
    }

    /// Returns the HNSW parameters for CPU execution.
    #[cfg(feature = "cpu")]
    pub(crate) const fn hnsw_params(&self) -> &HnswParams {
        &self.hnsw_params
    }
}
