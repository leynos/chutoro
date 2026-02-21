//! Hierarchical Navigable Small World (HNSW) build benchmarks.
//!
//! Measures the time to construct an HNSW index using both the plain
//! `build` path and the `build_with_edges` path that additionally
//! harvests candidate edges for MST construction.
use std::{path::PathBuf, time::Duration};

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_main, measurement::WallTime,
};

use chutoro_benches::{
    ef_sweep::make_hnsw_params_with_ef,
    error::BenchSetupError,
    params::HnswBenchParams,
    profiling::{
        EdgeScalingBounds, HnswMemoryInput, HnswMemoryRecord, ProfilingError,
        measure_peak_resident_set_size, write_hnsw_memory_report,
    },
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
const MAX_CONNECTIONS: &[usize] = &[8, 12, 16, 24];

/// Dataset size used for diverse synthetic pattern benchmarks.
const DIVERSE_POINT_COUNT: usize = 1_000;

/// Sampling cadence for peak resident-set-size profiling.
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_millis(2);

/// Report destination for derived memory metrics.
const MEMORY_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_memory_profile.csv"
);

/// Multiplicative edge-scaling tolerance around `expected = n * M`.
const EDGE_SCALING_BOUNDS: EdgeScalingBounds = EdgeScalingBounds::new(8, 8);

/// Creates a [`SyntheticSource`] with the given point count and the
/// module-level constants for dimensions and seed.
fn make_source(point_count: usize) -> Result<SyntheticSource, BenchSetupError> {
    Ok(SyntheticSource::generate(&SyntheticConfig {
        point_count,
        dimensions: DIMENSIONS,
        seed: SEED,
    })?)
}

/// Creates [`HnswParams`] for the given M value with `ef = M * 2`.
fn make_hnsw_params(m: usize) -> Result<HnswParams, BenchSetupError> {
    make_hnsw_params_with_ef(m, m.saturating_mul(2), SEED)
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

fn panic_on_bench_build_error<B>(result: Result<B, HnswError>, context: &str) {
    if let Err(err) = result {
        panic!("{context}: {err}");
    }
}

#[derive(Clone, Copy)]
struct SourceBenchSpec<'a> {
    bench_label: &'a str,
    fail_label: &'a str,
    point_count: usize,
}

fn bench_build_source<S: DataSource + Sync>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    spec: SourceBenchSpec<'_>,
    source: &S,
    params: &HnswParams,
) {
    let bench_params = HnswBenchParams {
        point_count: spec.point_count,
        max_connections: params.max_connections(),
        ef_construction: params.ef_construction(),
    };
    group.bench_with_input(
        BenchmarkId::new(spec.bench_label, &bench_params),
        &(source, params),
        |b, &(bench_source, input_params)| {
            b.iter_batched(
                || input_params.clone(),
                |cloned_params| {
                    panic_on_bench_build_error(
                        CpuHnsw::build(bench_source, cloned_params),
                        &format!("CpuHnsw::build failed for {}", spec.fail_label),
                    );
                },
                BatchSize::SmallInput,
            );
        },
    );
}

#[expect(
    clippy::excessive_nesting,
    reason = "Criterion bench_with_input + b.iter pattern requires deep nesting"
)]
fn bench_hnsw_build_generic<F>(
    c: &mut Criterion,
    group_name: &str,
    mut build_fn: F,
) -> Result<(), BenchSetupError>
where
    F: FnMut(&SyntheticSource, HnswParams) -> Result<(), HnswError>,
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
                |b, &(bench_source, input_params)| {
                    b.iter_batched(
                        || input_params.clone(),
                        |cloned_params| {
                            panic_on_bench_build_error(
                                build_fn(bench_source, cloned_params),
                                &format!("{group_name} failed during benchmark"),
                            );
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

fn hnsw_build_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    bench_hnsw_build_generic(c, "hnsw_build", |source, params| {
        CpuHnsw::build(source, params).map(|_| ())
    })
}

fn hnsw_build(c: &mut Criterion) {
    if let Err(err) = hnsw_build_impl(c) {
        panic!("hnsw_build benchmark setup failed: {err}");
    }
}

fn should_collect_memory_profile() -> bool {
    if let Ok(value) = std::env::var("CHUTORO_BENCH_HNSW_MEMORY_PROFILE") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "0" | "false" | "off") {
            return false;
        }
        if matches!(normalized.as_str(), "1" | "true" | "on") {
            return true;
        }
    }
    !std::env::args().any(|arg| arg == "--list" || arg == "--exact")
}

fn memory_report_path() -> PathBuf {
    std::env::var_os("CHUTORO_BENCH_HNSW_MEMORY_REPORT_PATH")
        .map_or_else(|| PathBuf::from(MEMORY_REPORT_PATH), PathBuf::from)
}

fn profile_hnsw_memory_impl() -> Result<Option<PathBuf>, BenchSetupError> {
    if !should_collect_memory_profile() {
        return Ok(None);
    }

    let report_path = memory_report_path();
    let mut records = Vec::new();

    for &point_count in POINT_COUNTS {
        let source = make_source(point_count)?;

        for &m in MAX_CONNECTIONS {
            let params = make_hnsw_params(m)?;
            let ef_construction = params.ef_construction();
            let (build_result, measurement) =
                match measure_peak_resident_set_size(MEMORY_SAMPLE_INTERVAL, || {
                    CpuHnsw::build_with_edges(&source, params.clone())
                }) {
                    Ok(measurement) => measurement,
                    Err(ProfilingError::UnsupportedPlatform { .. }) => return Ok(None),
                    Err(err) => return Err(err.into()),
                };
            let (_index, harvest) = build_result?;
            records.push(HnswMemoryRecord::new(
                HnswMemoryInput {
                    point_count,
                    max_connections: m,
                    ef_construction,
                    measurement,
                    edge_count: harvest.len(),
                },
                EDGE_SCALING_BOUNDS,
            )?);
        }
    }

    write_hnsw_memory_report(&report_path, &records)
        .map(Some)
        .map_err(BenchSetupError::from)
}

fn hnsw_build_with_edges_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let _maybe_report_path = profile_hnsw_memory_impl()?;
    bench_hnsw_build_generic(c, "hnsw_build_with_edges", |source, params| {
        CpuHnsw::build_with_edges(source, params).map(|_| ())
    })
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
    bench_build_source(
        &mut group,
        SourceBenchSpec {
            bench_label: "gaussian_blobs",
            fail_label: "gaussian source",
            point_count: DIVERSE_POINT_COUNT,
        },
        &gaussian,
        &params,
    );
    bench_build_source(
        &mut group,
        SourceBenchSpec {
            bench_label: "ring_manifold",
            fail_label: "ring source",
            point_count: DIVERSE_POINT_COUNT,
        },
        &ring,
        &params,
    );
    bench_build_source(
        &mut group,
        SourceBenchSpec {
            bench_label: "text_levenshtein",
            fail_label: "text source",
            point_count: DIVERSE_POINT_COUNT,
        },
        &text,
        &params,
    );

    if std::env::var("CHUTORO_BENCH_ENABLE_MNIST").as_deref() == Ok("1") {
        let mnist = SyntheticSource::load_mnist(&MnistConfig::default())?;
        bench_build_source(
            &mut group,
            SourceBenchSpec {
                bench_label: "mnist_baseline",
                fail_label: "MNIST source",
                point_count: mnist.len(),
            },
            &mnist,
            &params,
        );
    }

    group.finish();
    Ok(())
}

fn hnsw_build_diverse_sources(c: &mut Criterion) {
    if let Err(err) = hnsw_build_diverse_sources_impl(c) {
        panic!("hnsw_build_diverse_sources benchmark setup failed: {err}");
    }
}

mod bench_harness {
    use super::{hnsw_build, hnsw_build_diverse_sources, hnsw_build_with_edges};
    use criterion::criterion_group;

    criterion_group!(
        benches,
        hnsw_build,
        hnsw_build_with_edges,
        hnsw_build_diverse_sources
    );
}
criterion_main!(bench_harness::benches);
