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
        write!(
            f,
            "n={},M={},ef={}",
            self.point_count, self.max_connections, self.ef_construction,
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn hnsw_bench_params_display_includes_ef_construction() {
        let params = HnswBenchParams {
            point_count: 1_000,
            max_connections: 16,
            ef_construction: 32,
        };
        assert_eq!(params.to_string(), "n=1000,M=16,ef=32");
    }

    #[rstest]
    fn pipeline_bench_params_display() {
        let params = PipelineBenchParams { point_count: 500 };
        assert_eq!(params.to_string(), "n=500");
    }

    #[rstest]
    fn extraction_bench_params_display() {
        let params = ExtractionBenchParams {
            point_count: 100,
            min_cluster_size: 5,
        };
        assert_eq!(params.to_string(), "n=100,min=5");
    }
}
