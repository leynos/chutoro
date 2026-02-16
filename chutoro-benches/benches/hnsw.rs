//! Hierarchical Navigable Small World (HNSW) build benchmarks.
//!
//! Measures the time to construct an HNSW index using both the plain
//! `build` path and the `build_with_edges` path that additionally
//! harvests candidate edges for MST construction.
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

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
    error::BenchSetupError,
    params::HnswBenchParams,
    source::{SyntheticConfig, SyntheticSource},
};
use chutoro_core::{CpuHnsw, HnswParams};

/// Seed used for all synthetic data generation in this benchmark.
const SEED: u64 = 42;

/// Vector dimensionality for all benchmark datasets.
const DIMENSIONS: usize = 16;

/// Dataset sizes to benchmark.
const POINT_COUNTS: &[usize] = &[100, 500, 1_000, 5_000];

/// HNSW M parameter values to benchmark.
const MAX_CONNECTIONS: &[usize] = &[8, 16];

/// Creates a [`SyntheticSource`] with the given point count and the
/// module-level constants for dimensions and seed.
fn make_source(point_count: usize) -> Result<SyntheticSource, BenchSetupError> {
    Ok(SyntheticSource::generate(&SyntheticConfig {
        point_count,
        dimensions: DIMENSIONS,
        seed: SEED,
    })?)
}

/// Creates [`HnswParams`] for the given M value using the
/// module-level seed.
fn make_hnsw_params(m: usize) -> Result<HnswParams, BenchSetupError> {
    Ok(HnswParams::new(m, m.saturating_mul(2))?.with_rng_seed(SEED))
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "Criterion measurement closures cannot propagate errors via Result"
)]
fn hnsw_build_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let mut group = c.benchmark_group("hnsw_build");
    group.sample_size(10);

    for &point_count in POINT_COUNTS {
        let source = make_source(point_count)?;

        for &m in MAX_CONNECTIONS {
            let bench_params = HnswBenchParams {
                point_count,
                max_connections: m,
                ef_construction: m.saturating_mul(2),
            };
            let params = make_hnsw_params(m)?;

            group.bench_with_input(
                BenchmarkId::from_parameter(&bench_params),
                &(&source, &params),
                |b, &(source, params)| {
                    b.iter_batched(
                        || params.clone(),
                        |cloned_params| {
                            if let Err(err) = CpuHnsw::build(source, cloned_params) {
                                panic!("CpuHnsw::build failed during benchmark: {err}");
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
    Ok(())
}

fn hnsw_build(c: &mut Criterion) {
    if let Err(err) = hnsw_build_impl(c) {
        panic!("hnsw_build benchmark setup failed: {err}");
    }
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "Criterion measurement closures cannot propagate errors via Result"
)]
fn hnsw_build_with_edges_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let mut group = c.benchmark_group("hnsw_build_with_edges");
    group.sample_size(10);

    for &point_count in POINT_COUNTS {
        let source = make_source(point_count)?;

        for &m in MAX_CONNECTIONS {
            let bench_params = HnswBenchParams {
                point_count,
                max_connections: m,
                ef_construction: m.saturating_mul(2),
            };
            let params = make_hnsw_params(m)?;

            group.bench_with_input(
                BenchmarkId::from_parameter(&bench_params),
                &(&source, &params),
                |b, &(source, params)| {
                    b.iter_batched(
                        || params.clone(),
                        |cloned_params| {
                            if let Err(err) = CpuHnsw::build_with_edges(source, cloned_params) {
                                panic!("CpuHnsw::build_with_edges failed during benchmark: {err}");
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
    Ok(())
}

fn hnsw_build_with_edges(c: &mut Criterion) {
    if let Err(err) = hnsw_build_with_edges_impl(c) {
        panic!("hnsw_build_with_edges benchmark setup failed: {err}");
    }
}

criterion_group!(benches, hnsw_build, hnsw_build_with_edges);
criterion_main!(benches);
