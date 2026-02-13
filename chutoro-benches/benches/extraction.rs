//! Hierarchy extraction benchmarks.
//!
//! Measures the time to extract flat cluster labels from a minimum
//! spanning tree. This isolates the hierarchy extraction stage from
//! the preceding HNSW build and MST computation stages.
#![allow(missing_docs, reason = "Criterion macros generate undocumented items")]
#![allow(
    clippy::expect_used,
    reason = "benchmark setup is infallible for valid constants"
)]
#![allow(
    clippy::shadow_reuse,
    reason = "Criterion bench_with_input closures rebind parameter names"
)]
#![allow(
    clippy::excessive_nesting,
    reason = "Criterion bench_with_input + b.iter pattern requires deep nesting"
)]

use std::num::NonZeroUsize;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
    params::ExtractionBenchParams,
    source::{SyntheticConfig, SyntheticSource},
};
use chutoro_core::{
    CpuHnsw, HierarchyConfig, HnswParams, extract_labels_from_mst, parallel_kruskal,
};

/// Seed used for all synthetic data generation in this benchmark.
const SEED: u64 = 42;

/// Vector dimensionality for all benchmark datasets.
const DIMENSIONS: usize = 16;

/// Dataset sizes to benchmark.
const POINT_COUNTS: &[usize] = &[100, 500, 1_000];

/// Minimum cluster sizes to benchmark.
const MIN_CLUSTER_SIZES: &[usize] = &[5, 10];

/// HNSW M parameter used for edge generation.
const M: usize = 16;

fn extract_labels(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_labels");
    group.sample_size(20);

    for &point_count in POINT_COUNTS {
        let source = SyntheticSource::generate(&SyntheticConfig {
            point_count,
            dimensions: DIMENSIONS,
            seed: SEED,
        })
        .expect("synthetic source generation must succeed");

        let hnsw_params = HnswParams::new(M, M.saturating_mul(2))
            .expect("HNSW params must be valid")
            .with_rng_seed(SEED);

        let (_index, harvest) = CpuHnsw::build_with_edges(&source, hnsw_params)
            .expect("HNSW build_with_edges must succeed");

        let forest =
            parallel_kruskal(point_count, &harvest).expect("parallel_kruskal must succeed");

        let mst_edges = forest.edges();

        for &min_size in MIN_CLUSTER_SIZES {
            let bench_params = ExtractionBenchParams {
                point_count,
                min_cluster_size: min_size,
            };

            let config = HierarchyConfig::new(
                NonZeroUsize::new(min_size).expect("min_cluster_size must be non-zero"),
            );

            group.bench_with_input(
                BenchmarkId::from_parameter(&bench_params),
                &(point_count, mst_edges, &config),
                |b, &(node_count, edges, config)| {
                    b.iter(|| {
                        extract_labels_from_mst(node_count, edges, *config)
                            .expect("extraction must succeed");
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, extract_labels);
criterion_main!(benches);
