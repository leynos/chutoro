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

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};

use chutoro_benches::{
    error::BenchSetupError,
    params::HnswBenchParams,
    source::{
        Anisotropy, GaussianBlobConfig, ManifoldConfig, ManifoldPattern, MnistConfig,
        SyntheticConfig, SyntheticSource, SyntheticTextConfig,
    },
};
use chutoro_core::{CpuHnsw, DataSource, HnswError, HnswParams};

/// Seed used for all synthetic data generation in this benchmark.
const SEED: u64 = 42;

/// Vector dimensionality for all benchmark datasets.
const DIMENSIONS: usize = 16;

/// Dataset sizes to benchmark.
const POINT_COUNTS: &[usize] = &[100, 500, 1_000, 5_000];

/// HNSW M parameter values to benchmark.
const MAX_CONNECTIONS: &[usize] = &[8, 16];

/// Dataset size used for diverse synthetic pattern benchmarks.
const DIVERSE_POINT_COUNT: usize = 1_000;

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

fn make_gaussian_source() -> Result<SyntheticSource, BenchSetupError> {
    Ok(SyntheticSource::generate_gaussian_blobs(
        &GaussianBlobConfig {
            point_count: DIVERSE_POINT_COUNT,
            dimensions: DIMENSIONS,
            cluster_count: 8,
            separation: 6.0,
            anisotropy: Anisotropy::Isotropic(0.35),
            seed: SEED,
        },
    )?)
}

fn make_ring_source() -> Result<SyntheticSource, BenchSetupError> {
    Ok(SyntheticSource::generate_manifold(&ManifoldConfig {
        point_count: DIVERSE_POINT_COUNT,
        dimensions: DIMENSIONS,
        pattern: ManifoldPattern::Ring,
        major_radius: 7.5,
        thickness: 0.25,
        turns: 1,
        noise: 0.15,
        seed: SEED,
    })?)
}

fn make_text_source() -> Result<chutoro_benches::source::SyntheticTextSource, BenchSetupError> {
    Ok(SyntheticSource::generate_text(&SyntheticTextConfig {
        item_count: DIVERSE_POINT_COUNT,
        min_length: 6,
        max_length: 14,
        seed: SEED,
        alphabet: "acgtxyz".to_owned(),
        template_words: vec![
            "acgtacgt".to_owned(),
            "gattaca".to_owned(),
            "tgcactga".to_owned(),
        ],
        max_edits_per_item: 3,
    })?)
}

fn bench_build_numeric_source(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    source: &SyntheticSource,
    params: &HnswParams,
) {
    let bench_params = HnswBenchParams {
        point_count: DIVERSE_POINT_COUNT,
        max_connections: 16,
        ef_construction: 32,
    };
    group.bench_with_input(
        BenchmarkId::new(label, &bench_params),
        &(source, params),
        |b, &(source, params)| {
            b.iter_batched(
                || params.clone(),
                |cloned_params| {
                    if let Err(err) = CpuHnsw::build(source, cloned_params) {
                        panic!("CpuHnsw::build failed for {label} source: {err}");
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_build_text_source(
    group: &mut BenchmarkGroup<'_, WallTime>,
    source: &chutoro_benches::source::SyntheticTextSource,
    params: &HnswParams,
) {
    let bench_params = HnswBenchParams {
        point_count: DIVERSE_POINT_COUNT,
        max_connections: 16,
        ef_construction: 32,
    };
    group.bench_with_input(
        BenchmarkId::new("text_levenshtein", &bench_params),
        &(source, params),
        |b, &(source, params)| {
            b.iter_batched(
                || params.clone(),
                |cloned_params| {
                    if let Err(err) = CpuHnsw::build(source, cloned_params) {
                        panic!("CpuHnsw::build failed for text source: {err}");
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_build_mnist_source(
    group: &mut BenchmarkGroup<'_, WallTime>,
    source: &SyntheticSource,
    params: &HnswParams,
) {
    let bench_params = HnswBenchParams {
        point_count: source.len(),
        max_connections: 16,
        ef_construction: 32,
    };
    group.bench_with_input(
        BenchmarkId::new("mnist_baseline", &bench_params),
        &(source, params),
        |b, &(source, params)| {
            b.iter_batched(
                || params.clone(),
                |cloned_params| {
                    if let Err(err) = CpuHnsw::build(source, cloned_params) {
                        panic!("CpuHnsw::build failed for MNIST source: {err}");
                    }
                },
                BatchSize::SmallInput,
            );
        },
    );
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "Criterion measurement closures cannot propagate errors via Result"
)]
fn bench_hnsw_build_generic<B, F>(
    c: &mut Criterion,
    group_name: &str,
    mut build_fn: F,
) -> Result<(), BenchSetupError>
where
    F: FnMut(&SyntheticSource, HnswParams) -> Result<B, HnswError>,
{
    let mut group = c.benchmark_group(group_name);
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
                            if let Err(err) = build_fn(source, cloned_params) {
                                panic!("{group_name} failed during benchmark: {err}");
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

#[expect(
    clippy::panic_in_result_fn,
    reason = "Criterion measurement closures cannot propagate errors via Result"
)]
fn hnsw_build_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    bench_hnsw_build_generic(c, "hnsw_build", CpuHnsw::build)
        .inspect_err(|err| panic!("hnsw_build benchmark setup failed: {err}"))
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
    bench_hnsw_build_generic(c, "hnsw_build_with_edges", CpuHnsw::build_with_edges)
        .inspect_err(|err| panic!("hnsw_build_with_edges benchmark setup failed: {err}"))
}

fn hnsw_build_with_edges(c: &mut Criterion) {
    if let Err(err) = hnsw_build_with_edges_impl(c) {
        panic!("hnsw_build_with_edges benchmark setup failed: {err}");
    }
}

fn hnsw_build_diverse_sources_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let mut group = c.benchmark_group("hnsw_build_diverse_sources");
    group.sample_size(10);

    let params = make_hnsw_params(16)?;
    let gaussian = make_gaussian_source()?;
    let ring = make_ring_source()?;
    let text = make_text_source()?;
    bench_build_numeric_source(&mut group, "gaussian_blobs", &gaussian, &params);
    bench_build_numeric_source(&mut group, "ring_manifold", &ring, &params);
    bench_build_text_source(&mut group, &text, &params);

    if std::env::var("CHUTORO_BENCH_ENABLE_MNIST").as_deref() == Ok("1") {
        let mnist = SyntheticSource::load_mnist(&MnistConfig::default())?;
        bench_build_mnist_source(&mut group, &mnist, &params);
    }

    group.finish();
    Ok(())
}

fn hnsw_build_diverse_sources(c: &mut Criterion) {
    if let Err(err) = hnsw_build_diverse_sources_impl(c) {
        panic!("hnsw_build_diverse_sources benchmark setup failed: {err}");
    }
}

criterion_group!(
    benches,
    hnsw_build,
    hnsw_build_with_edges,
    hnsw_build_diverse_sources
);
criterion_main!(benches);
