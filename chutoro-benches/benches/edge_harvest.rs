//! Edge harvest construction benchmarks.
//!
//! Measures the cost of constructing an [`EdgeHarvest`] from a
//! pre-generated vector of candidate edges. This isolates the
//! sorting and canonicalization overhead from the HNSW insertion
//! work that produces the edges.
#![allow(missing_docs, reason = "Criterion macros generate undocumented items")]
#![allow(
    clippy::expect_used,
    reason = "benchmark setup is infallible for valid constants"
)]
#![allow(
    clippy::excessive_nesting,
    reason = "Criterion bench_with_input + b.iter pattern requires deep nesting"
)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
    params::PipelineBenchParams,
    source::{SyntheticConfig, SyntheticSource},
};
use chutoro_core::{CpuHnsw, EdgeHarvest, HnswParams};

/// Seed used for all synthetic data generation in this benchmark.
const SEED: u64 = 42;

/// Vector dimensionality for all benchmark datasets.
const DIMENSIONS: usize = 16;

/// Dataset sizes to benchmark.
const POINT_COUNTS: &[usize] = &[100, 500, 1_000];

/// HNSW M parameter used for edge generation.
const M: usize = 16;

fn edge_harvest_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_harvest_construction");
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

        // Build once to harvest edges for use in the benchmark loop.
        let (_index, harvest) = CpuHnsw::build_with_edges(&source, hnsw_params)
            .expect("HNSW build_with_edges must succeed");
        let raw_edges: Vec<_> = harvest.into_inner();

        let bench_params = PipelineBenchParams { point_count };

        group.bench_with_input(
            BenchmarkId::from_parameter(&bench_params),
            &raw_edges,
            |b, edges| {
                b.iter(|| {
                    let _harvest = EdgeHarvest::from_unsorted(edges.clone());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, edge_harvest_construction);
criterion_main!(benches);
