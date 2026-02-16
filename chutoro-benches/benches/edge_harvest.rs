//! Edge harvest construction benchmarks.
//!
//! Measures the cost of constructing an [`EdgeHarvest`] from a
//! pre-generated vector of candidate edges. This isolates the
//! sorting and canonicalization overhead from the HNSW insertion
//! work that produces the edges.
//!
//! Input edges are shuffled before benchmarking so that
//! `EdgeHarvest::from_unsorted` encounters a realistic unsorted
//! distribution rather than the pre-sorted output of
//! `build_with_edges`.
#![expect(
    missing_docs,
    reason = "Criterion macros generate items without doc comments"
)]
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use chutoro_benches::{
    error::BenchSetupError,
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

fn edge_harvest_construction_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let mut group = c.benchmark_group("edge_harvest_construction");
    group.sample_size(20);

    for &point_count in POINT_COUNTS {
        let source = SyntheticSource::generate(&SyntheticConfig {
            point_count,
            dimensions: DIMENSIONS,
            seed: SEED,
        })?;

        let hnsw_params = HnswParams::new(M, M.saturating_mul(2))?.with_rng_seed(SEED);

        // Build once to harvest edges for use in the benchmark loop.
        let (_index, harvest) = CpuHnsw::build_with_edges(&source, hnsw_params)?;
        let mut raw_edges: Vec<_> = harvest.into_inner();

        // Shuffle to present genuinely unsorted input — the harvest
        // output is already sorted so feeding it directly would
        // under-report the sorting and canonicalization cost.
        let mut rng = SmallRng::seed_from_u64(SEED);
        raw_edges.shuffle(&mut rng);

        let bench_params = PipelineBenchParams { point_count };

        group.bench_with_input(
            BenchmarkId::from_parameter(&bench_params),
            &raw_edges,
            |b, edges| {
                b.iter_batched(
                    || edges.clone(),
                    |cloned_edges| {
                        let _harvest = EdgeHarvest::from_unsorted(cloned_edges);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
    Ok(())
}

fn edge_harvest_construction(c: &mut Criterion) {
    if let Err(err) = edge_harvest_construction_impl(c) {
        panic!("edge_harvest_construction benchmark setup failed: {err}");
    }
}

criterion_group!(benches, edge_harvest_construction);
criterion_main!(benches);
