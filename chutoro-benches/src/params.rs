//! Benchmark parameter types.
//!
//! Groups related benchmark parameters into structs so that benchmark
//! helper functions stay under the Clippy `too-many-arguments` threshold.

use std::fmt;

/// Parameters for an HNSW benchmark run.
#[derive(Clone, Debug)]
pub struct HnswBenchParams {
    /// Number of points in the dataset.
    pub point_count: usize,
    /// HNSW maximum connections per node (M).
    pub max_connections: usize,
    /// HNSW search width during construction.
    pub ef_construction: usize,
}

impl fmt::Display for HnswBenchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n={},M={}", self.point_count, self.max_connections,)
    }
}

/// Parameters for an MST or extraction benchmark run.
#[derive(Clone, Debug)]
pub struct PipelineBenchParams {
    /// Number of points in the dataset.
    pub point_count: usize,
}

impl fmt::Display for PipelineBenchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n={}", self.point_count)
    }
}

/// Parameters for a hierarchy extraction benchmark run.
#[derive(Clone, Debug)]
pub struct ExtractionBenchParams {
    /// Number of points in the dataset.
    pub point_count: usize,
    /// Minimum cluster size for hierarchy extraction.
    pub min_cluster_size: usize,
}

impl fmt::Display for ExtractionBenchParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n={},min={}", self.point_count, self.min_cluster_size,)
    }
}
