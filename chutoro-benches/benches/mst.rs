//! Minimum spanning tree (MST) parallel Kruskal benchmarks.
//!
//! Measures the time to compute a minimum spanning forest from an
//! edge harvest produced by HNSW construction. This isolates the
//! MST computation from the preceding HNSW build and edge harvest
//! stages.
#![expect(
    missing_docs,
    reason = "Criterion macros generate items without doc comments"
)]
#![expect(
    clippy::shadow_reuse,
    reason = "Criterion bench_with_input closures rebind parameter names"
)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
    error::BenchSetupError,
    params::PipelineBenchParams,
    source::{SyntheticConfig, SyntheticSource},
};
use chutoro_core::{CpuHnsw, HnswParams, parallel_kruskal};

/// Seed used for all synthetic data generation in this benchmark.
const SEED: u64 = 42;

/// Vector dimensionality for all benchmark datasets.
const DIMENSIONS: usize = 16;

/// Dataset sizes to benchmark.
const POINT_COUNTS: &[usize] = &[100, 500, 1_000];

/// HNSW M parameter used for edge generation.
const M: usize = 16;

fn mst_parallel_kruskal_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let mut group = c.benchmark_group("parallel_kruskal");
    group.sample_size(20);

    for &point_count in POINT_COUNTS {
        let source = SyntheticSource::generate(&SyntheticConfig {
            point_count,
            dimensions: DIMENSIONS,
            seed: SEED,
        })?;

        let hnsw_params = HnswParams::new(M, M.saturating_mul(2))?.with_rng_seed(SEED);

        let (_index, harvest) = CpuHnsw::build_with_edges(&source, hnsw_params)?;

        let bench_params = PipelineBenchParams { point_count };

        group.bench_with_input(
            BenchmarkId::from_parameter(&bench_params),
            &(point_count, &harvest),
            |b, &(node_count, harvest)| {
                b.iter(|| {
                    let _forest = parallel_kruskal(node_count, harvest);
                });
            },
        );
    }

    group.finish();
    Ok(())
}

fn mst_parallel_kruskal(c: &mut Criterion) {
    if let Err(err) = mst_parallel_kruskal_impl(c) {
        panic!("mst_parallel_kruskal benchmark setup failed: {err}");
    }
}

criterion_group!(benches, mst_parallel_kruskal);
criterion_main!(benches);
