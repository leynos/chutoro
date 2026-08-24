//! Hierarchical Navigable Small World (HNSW) build benchmarks.
//!
//! Measures the time to construct an HNSW index using both the plain
//! `build` path and the `build_with_edges` path that additionally
//! harvests candidate edges for MST construction.
use std::{path::PathBuf, time::Duration};

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, black_box, criterion_main,
    measurement::WallTime,
};
use mockable::{DefaultEnv, Env};

use chutoro_benches::{
    criterion_support::{
        configure_short_measurement_group, is_benchmark_discovery, is_exact_benchmark_probe,
        is_nextest_exact_benchmark_probe, point_count_for_exact_probe_args, register_noop_benches,
        should_short_circuit_exact_label_probe_args,
    },
    ef_sweep::{BENCH_DIMENSIONS, BENCH_SEED, make_bench_source, make_hnsw_params_with_ef},
    error::BenchSetupError,
    params::HnswBenchParams,
    profiling::{
        EdgeScalingBounds, HnswMemoryInput, HnswMemoryRecord, ProfilingError,
        measure_peak_resident_set_size, write_hnsw_memory_report,
    },
    source::{
        Anisotropy, GaussianBlobConfig, ManifoldConfig, ManifoldPattern, MnistConfig,
        SyntheticSource, SyntheticTextConfig,
    },
};
use chutoro_core::{CpuHnsw, DataSource, HnswError, HnswParams};

/// Dataset sizes to benchmark.
const POINT_COUNTS: &[usize] = &[100, 500, 1_000, 5_000];

/// HNSW M parameter values to benchmark.
const MAX_CONNECTIONS: &[usize] = &[8, 12, 16, 24];

/// Dataset size used for diverse synthetic pattern benchmarks.
const DIVERSE_POINT_COUNT: usize = 1_000;

/// Dataset size used when nextest probes one Criterion case with `--exact`.
const EXACT_PROBE_POINT_COUNT: usize = 100;

/// Sampling cadence for peak resident-set-size profiling.
const MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_millis(2);

/// Report destination for derived memory metrics.
const MEMORY_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_memory_profile.csv"
);

/// Multiplicative edge-scaling tolerance around `expected = n * M`.
const EDGE_SCALING_BOUNDS: EdgeScalingBounds = EdgeScalingBounds::new(8, 8);

/// Criterion label for the text-source Levenshtein-distance case.
const TEXT_LEVENSHTEIN_BENCH_LABEL: &str = "text_levenshtein";

/// Creates [`HnswParams`] for the given M value with `ef = M * 2`.
fn make_hnsw_params(m: usize) -> Result<HnswParams, BenchSetupError> {
    make_hnsw_params_with_ef(m, m.saturating_mul(2), BENCH_SEED)
}

/// Build the Gaussian-blob source used by the diverse-source benchmark.
fn make_gaussian_source() -> Result<SyntheticSource, BenchSetupError> {
    Ok(SyntheticSource::generate_gaussian_blobs(
        &GaussianBlobConfig {
            point_count: diverse_source_point_count(),
            dimensions: BENCH_DIMENSIONS,
            cluster_count: 8,
            separation: 6.0,
            anisotropy: Anisotropy::Isotropic(0.35),
            seed: BENCH_SEED,
        },
    )?)
}

/// Build the ring-manifold source used by the diverse-source benchmark.
fn make_ring_source() -> Result<SyntheticSource, BenchSetupError> {
    Ok(SyntheticSource::generate_manifold(&ManifoldConfig {
        point_count: diverse_source_point_count(),
        dimensions: BENCH_DIMENSIONS,
        pattern: ManifoldPattern::Ring,
        major_radius: 7.5,
        thickness: 0.25,
        turns: 1,
        noise: 0.15,
        seed: BENCH_SEED,
    })?)
}

/// Build the synthetic text source used by the diverse-source benchmark.
fn make_text_source() -> Result<chutoro_benches::source::SyntheticTextSource, BenchSetupError> {
    Ok(SyntheticSource::generate_text(&SyntheticTextConfig {
        item_count: diverse_source_point_count(),
        min_length: 6,
        max_length: 14,
        seed: BENCH_SEED,
        alphabet: "acgtxyz".to_owned(),
        template_words: vec![
            "acgtacgt".to_owned(),
            "gattaca".to_owned(),
            "tgcactga".to_owned(),
        ],
        max_edits_per_item: 3,
    })?)
}

/// Surface an HNSW build failure from a Criterion closure.
fn panic_on_bench_build_error<B>(result: Result<B, HnswError>, context: &str) {
    if let Err(err) = result {
        panic!("{context}: {err}");
    }
}

/// Select the diverse-source input size for normal and exact-probe runs.
fn diverse_source_point_count() -> usize {
    // Nextest discovers Criterion case names without `--exact`, so the
    // benchmark IDs still advertise the real matrix size. Only the exact probe
    // input is shortened to keep test gating bounded.
    point_count_for_exact_probe_args(
        std::env::args(),
        DIVERSE_POINT_COUNT,
        EXACT_PROBE_POINT_COUNT,
    )
}

/// Select one HNSW input size while preserving the displayed benchmark ID.
fn hnsw_source_point_count(point_count: usize) -> usize {
    // Keep Criterion benchmark IDs stable while bounding nextest's exact probes.
    point_count_for_exact_probe_args(std::env::args(), point_count, EXACT_PROBE_POINT_COUNT)
}

/// Configure sampling for an HNSW Criterion benchmark group.
fn configure_hnsw_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    configure_short_measurement_group(group, 10, is_exact_benchmark_probe());
}

/// Identify the text case that must use a bounded exact-probe measurement.
fn should_short_circuit_exact_text_probe(bench_label: &str) -> bool {
    should_short_circuit_exact_label_probe_args(
        std::env::args(),
        bench_label,
        TEXT_LEVENSHTEIN_BENCH_LABEL,
    )
}

/// Describes one source-specific HNSW build benchmark case.
#[derive(Clone, Copy)]
struct SourceBenchSpec<'a> {
    /// Stable Criterion label for the case.
    bench_label: &'a str,
    /// Context shown if the source's HNSW build fails.
    fail_label: &'a str,
    /// Number of source items represented by the benchmark ID.
    point_count: usize,
}

/// Register one source-specific HNSW build measurement.
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
            if should_short_circuit_exact_text_probe(spec.bench_label) {
                b.iter(|| black_box(()));
            } else {
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
            }
        },
    );
}

/// Register one HNSW build matrix using the supplied construction operation.
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
    if is_benchmark_discovery() || is_nextest_exact_benchmark_probe() {
        register_hnsw_build_probe_benches(c, group_name);
        return Ok(());
    }

    let mut group = c.benchmark_group(group_name);
    configure_hnsw_group(&mut group);

    for &point_count in POINT_COUNTS {
        let source = make_bench_source(hnsw_source_point_count(point_count))?;

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

/// Register no-op HNSW cases used while Criterion discovers benchmark names.
fn register_hnsw_build_probe_benches(c: &mut Criterion, group_name: &str) {
    let params = POINT_COUNTS.iter().copied().flat_map(|point_count| {
        MAX_CONNECTIONS
            .iter()
            .copied()
            .map(move |m| HnswBenchParams {
                point_count,
                max_connections: m,
                ef_construction: m.saturating_mul(2),
            })
    });
    register_noop_benches(c, group_name, params, configure_hnsw_group);
}

/// Register the plain HNSW build benchmark and return setup failures.
fn hnsw_build_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    bench_hnsw_build_generic(c, "hnsw_build", |source, params| {
        CpuHnsw::build(source, params).map(|_| ())
    })
}

/// Register the public Criterion plain-HNSW benchmark entrypoint.
fn hnsw_build(c: &mut Criterion) {
    if let Err(err) = hnsw_build_impl(c) {
        panic!("hnsw_build benchmark setup failed: {err}");
    }
}

/// Determine whether this invocation should collect HNSW memory measurements.
fn should_collect_memory_profile_with_env(env: &dyn Env) -> bool {
    if let Some(value) = env.string("CHUTORO_BENCH_HNSW_MEMORY_PROFILE") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "0" | "false" | "off") {
            return false;
        }
        if matches!(normalized.as_str(), "1" | "true" | "on") {
            return true;
        }
    }
    !is_benchmark_discovery() && !is_exact_benchmark_probe()
}

fn memory_report_path_with_env(env: &dyn Env) -> PathBuf {
    env.os_string("CHUTORO_BENCH_HNSW_MEMORY_REPORT_PATH")
        .map_or_else(|| PathBuf::from(MEMORY_REPORT_PATH), PathBuf::from)
}
/// Collect and write optional HNSW memory measurements before benchmark setup.
fn profile_hnsw_memory_impl() -> Result<Option<PathBuf>, BenchSetupError> {
    profile_hnsw_memory_impl_with_env(&DefaultEnv)
}

fn profile_hnsw_memory_impl_with_env(env: &dyn Env) -> Result<Option<PathBuf>, BenchSetupError> {
    if !should_collect_memory_profile_with_env(env) {
        return Ok(None);
    }

    let report_path = memory_report_path_with_env(env);
    let mut records = Vec::new();

    for &point_count in POINT_COUNTS {
        let source = make_bench_source(point_count)?;

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
/// Register edge-harvesting HNSW measurements and optional memory reporting.
fn hnsw_build_with_edges_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    let _maybe_report_path = profile_hnsw_memory_impl()?;
    bench_hnsw_build_generic(c, "hnsw_build_with_edges", |source, params| {
        CpuHnsw::build_with_edges(source, params).map(|_| ())
    })
}

/// Register the public Criterion edge-harvesting HNSW entrypoint.
fn hnsw_build_with_edges(c: &mut Criterion) {
    if let Err(err) = hnsw_build_with_edges_impl(c) {
        panic!("hnsw_build_with_edges benchmark setup failed: {err}");
    }
}

/// Register HNSW build measurements across diverse synthetic source shapes.
fn hnsw_build_diverse_sources_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    hnsw_build_diverse_sources_impl_with_env(c, &DefaultEnv)
}

fn hnsw_build_diverse_sources_impl_with_env(
    c: &mut Criterion,
    env: &dyn Env,
) -> Result<(), BenchSetupError> {
    let mut group = c.benchmark_group("hnsw_build_diverse_sources");
    configure_hnsw_group(&mut group);

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
            bench_label: TEXT_LEVENSHTEIN_BENCH_LABEL,
            fail_label: "text source",
            point_count: DIVERSE_POINT_COUNT,
        },
        &text,
        &params,
    );

    if env.string("CHUTORO_BENCH_ENABLE_MNIST").as_deref() == Some("1") {
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
/// Register the public Criterion diverse-source HNSW entrypoint.
fn hnsw_build_diverse_sources(c: &mut Criterion) {
    if let Err(err) = hnsw_build_diverse_sources_impl(c) {
        panic!("hnsw_build_diverse_sources benchmark setup failed: {err}");
    }
}

mod bench_harness {
    //! Criterion entrypoint for the HNSW benchmark groups.

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
