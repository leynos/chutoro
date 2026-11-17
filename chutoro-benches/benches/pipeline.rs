//! Whole CPU clustering pipeline benchmarks.
//!
//! Measures `Chutoro::run` end to end for the two dataset sizes the
//! core-distance stage is specified against. The HNSW build, core-distance,
//! mutual-reachability, and Kruskal stages all run inside the timed closure, so
//! a change to one stage is visible as a change to the whole pipeline.
//!
//! With the `pipeline-profile` feature, one untimed run per size also writes
//! the per-stage breakdown before Criterion registers its benchmarks. The
//! profile is written by `chutoro_benches::pipeline_profile`.
#![expect(
    missing_docs,
    reason = "Criterion macros generate items without doc comments"
)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use chutoro_benches::{
    criterion_support::{
        configure_short_measurement_group, is_benchmark_discovery, is_exact_benchmark_probe,
        is_nextest_exact_benchmark_probe, point_count_for_exact_probe_args, register_noop_benches,
    },
    error::BenchSetupError,
    params::PipelineBenchParams,
    source::{SyntheticConfig, SyntheticSource},
};
use chutoro_core::{Chutoro, ChutoroBuilder, ExecutionStrategy, HnswParams};

/// Seed used for synthetic data generation and HNSW level sampling.
const SEED: u64 = 42;

/// Vector dimensionality for the benchmark datasets.
const DIMENSIONS: usize = 16;

/// Dataset sizes the core-distance change is measured against.
const POINT_COUNTS: &[usize] = &[1_000, 10_000];

/// HNSW M parameter used for every case.
const MAX_CONNECTIONS: usize = 16;

/// HNSW construction search width used for every case.
const EF_CONSTRUCTION: usize = 32;

/// Minimum cluster size used for every case.
const MIN_CLUSTER_SIZE: usize = 5;

/// Dataset size used when nextest probes one Criterion case with `--exact`.
const EXACT_PROBE_POINT_COUNT: usize = 100;

/// One prepared benchmark case: its reported size and its timed inputs.
struct PipelineCase {
    /// Point count shown in the Criterion benchmark ID.
    point_count: usize,
    /// Source generated once, outside the timed closure.
    source: SyntheticSource,
    /// Configured pipeline reused across iterations.
    chutoro: Chutoro,
}

/// Builds every benchmark case, generating data outside the timed closures.
fn build_cases() -> Result<Vec<PipelineCase>, BenchSetupError> {
    let params = HnswParams::new(MAX_CONNECTIONS, EF_CONSTRUCTION)?.with_rng_seed(SEED);
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(MIN_CLUSTER_SIZE)
        .with_execution_strategy(ExecutionStrategy::CpuOnly)
        .with_hnsw_params(params)
        .build()?;

    POINT_COUNTS
        .iter()
        .copied()
        .map(|point_count| {
            let source_point_count = point_count_for_exact_probe_args(
                std::env::args(),
                point_count,
                EXACT_PROBE_POINT_COUNT,
            );
            let source = SyntheticSource::generate(&SyntheticConfig {
                point_count: source_point_count,
                dimensions: DIMENSIONS,
                seed: SEED,
            })?;
            Ok(PipelineCase {
                point_count,
                source,
                chutoro: chutoro.clone(),
            })
        })
        .collect()
}

/// Registers the whole-pipeline measurements and returns setup failures.
fn pipeline_impl(c: &mut Criterion) -> Result<(), BenchSetupError> {
    if is_benchmark_discovery() || is_nextest_exact_benchmark_probe() {
        register_pipeline_probe_benches(c);
        return Ok(());
    }

    let cases = build_cases()?;
    profile_cases(&cases)?;

    let mut group = c.benchmark_group("pipeline");
    configure_short_measurement_group(&mut group, 10, is_exact_benchmark_probe());

    for case in &cases {
        let bench_params = PipelineBenchParams {
            point_count: case.point_count,
        };
        group.bench_with_input(
            BenchmarkId::from_parameter(&bench_params),
            case,
            |b, case| {
                b.iter(|| {
                    if let Err(err) = case.chutoro.run(&case.source) {
                        panic!("whole pipeline failed during benchmark: {err}");
                    }
                });
            },
        );
    }

    group.finish();
    Ok(())
}

/// Writes the optional stage profile before Criterion registration.
#[cfg(feature = "pipeline-profile")]
fn profile_cases(cases: &[PipelineCase]) -> Result<(), BenchSetupError> {
    use chutoro_benches::pipeline_profile::{measure_pipeline_stages, write_pipeline_profile};

    let samples = cases
        .iter()
        .map(|case| measure_pipeline_stages(case.point_count, &case.source, &case.chutoro))
        .collect::<Result<Vec<_>, _>>()?;
    let path = write_pipeline_profile(&samples)?;
    eprintln!("[pipeline-profile] stage profile written to {path}");
    Ok(())
}

/// Skips profiling when the `pipeline-profile` feature is not compiled in.
#[cfg(not(feature = "pipeline-profile"))]
fn profile_cases(_cases: &[PipelineCase]) -> Result<(), BenchSetupError> {
    Ok(())
}

/// Registers no-op pipeline cases used while Criterion discovers names.
fn register_pipeline_probe_benches(c: &mut Criterion) {
    let params = POINT_COUNTS
        .iter()
        .copied()
        .map(|point_count| PipelineBenchParams { point_count });
    register_noop_benches(c, "pipeline", params, |group| {
        configure_short_measurement_group(group, 10, is_exact_benchmark_probe());
    });
}

/// Registers the public Criterion whole-pipeline entrypoint.
fn pipeline(c: &mut Criterion) {
    if let Err(err) = pipeline_impl(c) {
        panic!("pipeline benchmark setup failed: {err}");
    }
}

criterion_group!(benches, pipeline);
criterion_main!(benches);
