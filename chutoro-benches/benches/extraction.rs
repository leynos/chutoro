//! Hierarchy extraction benchmarks.
//!
//! Measures the time to extract flat cluster labels from a minimum
//! spanning tree. This isolates the hierarchy extraction stage from
//! the preceding HNSW build and MST computation stages.
#![expect(
    missing_docs,
    reason = "Criterion macros generate items without doc comments"
)]
#![expect(
    clippy::shadow_reuse,
    reason = "Criterion bench_with_input closures rebind parameter names"
)]
#![expect(
    clippy::excessive_nesting,
    reason = "Criterion bench_with_input + b.iter pattern requires deep nesting"
)]

use std::num::NonZeroUsize;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
    error::BenchSetupError,
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

#[expect(
    clippy::panic_in_result_fn,
    reason = "Criterion measurement closures cannot propagate errors via Result"
)]
fn extract_labels_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let mut group = c.benchmark_group("extract_labels");
    group.sample_size(20);

    for &point_count in POINT_COUNTS {
        let source = SyntheticSource::generate(&SyntheticConfig {
            point_count,
            dimensions: DIMENSIONS,
            seed: SEED,
        })?;

        let hnsw_params = HnswParams::new(M, M.saturating_mul(2))?.with_rng_seed(SEED);

        let (_index, harvest) = CpuHnsw::build_with_edges(&source, hnsw_params)?;

        let forest = parallel_kruskal(point_count, &harvest)?;

        let mst_edges = forest.edges();

        for &min_size in MIN_CLUSTER_SIZES {
            let bench_params = ExtractionBenchParams {
                point_count,
                min_cluster_size: min_size,
            };

            let min_cluster = NonZeroUsize::new(min_size).ok_or(BenchSetupError::ZeroValue {
                context: "min_cluster_size",
            })?;
            let config = HierarchyConfig::new(min_cluster);

            group.bench_with_input(
                BenchmarkId::from_parameter(&bench_params),
                &(point_count, mst_edges, &config),
                |b, &(node_count, edges, config)| {
                    b.iter(|| {
                        if let Err(err) = extract_labels_from_mst(node_count, edges, *config) {
                            panic!("extract_labels_from_mst failed during benchmark: {err}");
                        }
                    });
                },
            );
        }
    }

    group.finish();
    Ok(())
}

fn extract_labels(c: &mut Criterion) {
    if let Err(err) = extract_labels_impl(c) {
        panic!("extract_labels benchmark setup failed: {err}");
    }
}

criterion_group!(benches, extract_labels);
criterion_main!(benches);
