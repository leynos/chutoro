//! HNSW build benchmarks.
//!
//! Measures the time to construct an HNSW index using both the plain
//! `build` path and the `build_with_edges` path that additionally
//! harvests candidate edges for MST construction.
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

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
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

fn hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    group.sample_size(10);

    for &point_count in POINT_COUNTS {
        let source = SyntheticSource::generate(&SyntheticConfig {
            point_count,
            dimensions: DIMENSIONS,
            seed: SEED,
        })
        .expect("synthetic source generation must succeed");

        for &m in MAX_CONNECTIONS {
            let bench_params = HnswBenchParams {
                point_count,
                max_connections: m,
                ef_construction: m.saturating_mul(2),
            };
            let hnsw_params = HnswParams::new(m, bench_params.ef_construction)
                .expect("HNSW params must be valid")
                .with_rng_seed(SEED);

            group.bench_with_input(
                BenchmarkId::from_parameter(&bench_params),
                &(&source, &hnsw_params),
                |b, &(source, params)| {
                    b.iter(|| {
                        CpuHnsw::build(source, params.clone()).expect("HNSW build must succeed");
                    });
                },
            );
        }
    }

    group.finish();
}

fn hnsw_build_with_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build_with_edges");
    group.sample_size(10);

    for &point_count in POINT_COUNTS {
        let source = SyntheticSource::generate(&SyntheticConfig {
            point_count,
            dimensions: DIMENSIONS,
            seed: SEED,
        })
        .expect("synthetic source generation must succeed");

        for &m in MAX_CONNECTIONS {
            let bench_params = HnswBenchParams {
                point_count,
                max_connections: m,
                ef_construction: m.saturating_mul(2),
            };
            let hnsw_params = HnswParams::new(m, bench_params.ef_construction)
                .expect("HNSW params must be valid")
                .with_rng_seed(SEED);

            group.bench_with_input(
                BenchmarkId::from_parameter(&bench_params),
                &(&source, &hnsw_params),
                |b, &(source, params)| {
                    b.iter(|| {
                        CpuHnsw::build_with_edges(source, params.clone())
                            .expect("HNSW build_with_edges must succeed");
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, hnsw_build, hnsw_build_with_edges);
criterion_main!(benches);
