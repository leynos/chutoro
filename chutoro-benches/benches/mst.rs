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
#![expect(
    clippy::excessive_nesting,
    reason = "Criterion bench_with_input + error handling requires deep nesting"
)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
    criterion_support::{
        configure_short_measurement_group, is_benchmark_discovery, is_exact_benchmark_probe,
        is_nextest_exact_benchmark_probe, register_noop_benches,
    },
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

#[expect(
    clippy::panic_in_result_fn,
    reason = "Criterion measurement closures cannot propagate errors via Result"
)]
fn mst_parallel_kruskal_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    if is_benchmark_discovery() || is_nextest_exact_benchmark_probe() {
        register_mst_discovery_benches(c);
        return Ok(());
    }

    let mut group = c.benchmark_group("parallel_kruskal");
    configure_short_measurement_group(&mut group, 20, is_exact_benchmark_probe());

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
                    if let Err(err) = parallel_kruskal(node_count, harvest) {
                        panic!("parallel_kruskal failed during benchmark: {err}");
                    }
                });
            },
        );
    }

    group.finish();
    Ok(())
}

fn register_mst_discovery_benches(c: &mut Criterion) {
    let params = POINT_COUNTS
        .iter()
        .copied()
        .map(|point_count| PipelineBenchParams { point_count });
    register_noop_benches(c, "parallel_kruskal", params, |_| {});
}

fn mst_parallel_kruskal(c: &mut Criterion) {
    if let Err(err) = mst_parallel_kruskal_impl(c) {
        panic!("mst_parallel_kruskal benchmark setup failed: {err}");
    }
}

criterion_group!(benches, mst_parallel_kruskal);
criterion_main!(benches);
