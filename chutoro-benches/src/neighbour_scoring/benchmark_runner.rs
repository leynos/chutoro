//! Criterion runner for HNSW neighbour-scoring evaluation.
//!
//! The harness measures the query-centric dense-provider path at realistic
//! HNSW candidate bucket sizes and emits small diagnostic CSV reports under
//! `target/benchmarks/` so roadmap item 2.3.1 can be closed on evidence.

use crate::{
    criterion_support::configure_short_measurement_group,
    neighbour_scoring::{
        BUILD_PROFILE_ENV, ReportTarget, build_profile_report_target_value, truthy_env_value,
    },
};
use camino::Utf8Path;
use chutoro_core::DataSource;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, black_box, measurement::WallTime,
};
use mockable::{DefaultEnv, Env};

use super::{
    CandidateBucket, ScoringFixture, benchmark_support::BenchError, benchmark_support::BenchResult,
    build_profile::report_parent_dir_with_env, make_fixture, scoring_plan,
    write_build_profile_report, write_lane_utilisation_report,
};

/// Query row used for every neighbour-scoring benchmark iteration.
const QUERY_INDEX: usize = 0;
/// Environment variable enabling shorter benchmark measurements.
const SHORT_MEASUREMENT_ENV: &str = "CHUTORO_BENCH_NEIGHBOUR_SHORT_MEASUREMENT";

/// Apply the group's optional short-measurement configuration.
fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    configure_short_measurement_group(group, 10, should_use_short_measurement());
}

/// Interpret the short-measurement environment variable value.
fn should_use_short_measurement_value(value: Option<&str>) -> bool {
    truthy_env_value(value)
}

/// Read whether short benchmark measurements are enabled.
fn should_use_short_measurement() -> bool {
    should_use_short_measurement_with_env(&DefaultEnv)
}

/// Read short-measurement configuration through an injected environment reader.
fn should_use_short_measurement_with_env(env: &dyn Env) -> bool {
    should_use_short_measurement_value(env.string(SHORT_MEASUREMENT_ENV).as_deref())
}
/// Measure distances from the benchmark query to selected candidates.
fn score_candidates(
    scoring_fixture: &ScoringFixture,
    candidates: &[usize],
) -> Result<Vec<f32>, chutoro_core::DataSourceError> {
    scoring_fixture
        .provider
        .batch_distances(black_box(QUERY_INDEX), black_box(candidates))
}

/// Convert a candidate bucket to Criterion throughput metadata.
fn throughput_for(bucket: CandidateBucket) -> BenchResult<Throughput> {
    let throughput =
        u64::try_from(bucket.size()).map_err(|source| BenchError::CandidateCountConversion {
            candidate_count: bucket.size(),
            source,
        })?;
    Ok(Throughput::Elements(throughput))
}

/// Build a stable Criterion identifier for one benchmark case.
fn bench_id_for(bucket: CandidateBucket, dimension: usize) -> BenchmarkId {
    BenchmarkId::new(
        bucket.kind_name(),
        format!("dim_{dimension}_candidates_{}", bucket.size()),
    )
}

/// Execute and black-box a single scoring iteration.
fn run_scoring_iteration(scoring_fixture: &ScoringFixture, candidates: &[usize]) {
    match score_candidates(scoring_fixture, candidates) {
        Ok(distances) => {
            black_box(distances);
        }
        Err(err) => {
            panic!("neighbour_scoring benchmark iteration failed: {err}");
        }
    }
}

/// Register one dimension and candidate-bucket benchmark case.
fn bench_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    bucket: CandidateBucket,
) -> BenchResult<()> {
    let fixture = make_fixture(dimension, bucket.size())?;
    score_candidates(&fixture, &fixture.candidates)?;
    group.throughput(throughput_for(bucket)?);
    let id = bench_id_for(bucket, dimension);
    group.bench_with_input(id, &fixture, |b, scoring_fixture| {
        b.iter(|| run_scoring_iteration(scoring_fixture, &scoring_fixture.candidates));
    });
    Ok(())
}

/// Run the neighbour-scoring benchmark with production report writers.
fn neighbour_scoring_impl(c: &mut Criterion) -> BenchResult<()> {
    neighbour_scoring_impl_with(
        c,
        &DefaultEnv,
        (
            |report_parent_dir| write_lane_utilisation_report(report_parent_dir).map(drop),
            |report_parent_dir| write_build_profile_report(report_parent_dir).map(drop),
        ),
        bench_case,
    )
}

/// Run the benchmark with injected report writers and case registration.
fn neighbour_scoring_impl_with(
    c: &mut Criterion,
    env: &dyn Env,
    report_writers: (
        impl FnOnce(&Utf8Path) -> BenchResult<()>,
        impl FnOnce(Option<&Utf8Path>) -> BenchResult<()>,
    ),
    mut scoring_case: impl FnMut(
        &mut BenchmarkGroup<'_, WallTime>,
        usize,
        CandidateBucket,
    ) -> BenchResult<()>,
) -> BenchResult<()> {
    let report_parent_dir = report_parent_dir_with_env(env);
    let build_profile_target = build_profile_report_target_value(
        env.string(BUILD_PROFILE_ENV).as_deref(),
        &report_parent_dir,
    );
    let build_profile_report_dir = build_profile_target
        .as_ref()
        .map(ReportTarget::report_parent_dir);
    let (lane_report_writer, build_profile_writer) = report_writers;
    lane_report_writer(&report_parent_dir)?;
    build_profile_writer(build_profile_report_dir)?;
    let mut group = c.benchmark_group("neighbour_scoring");
    configure_group(&mut group);
    for (dimension, bucket) in scoring_plan() {
        scoring_case(&mut group, dimension, bucket)?;
    }
    group.finish();
    Ok(())
}

/// Registers the neighbour-scoring benchmark group and its diagnostic reports.
///
/// This creates the `neighbour_scoring` Criterion group and writes its
/// lane-utilisation and optional build-profile CSV diagnostics before
/// registering benchmark cases.
///
/// # Examples
///
/// ```no_run
/// use criterion::Criterion;
/// use chutoro_benches::neighbour_scoring::run_neighbour_scoring;
///
/// let mut criterion = Criterion::default();
/// run_neighbour_scoring(&mut criterion);
/// ```
///
/// # Panics
///
/// Panics when fixture construction, diagnostic report generation, or
/// benchmark registration fails.
pub fn neighbour_scoring(c: &mut Criterion) {
    if let Err(err) = neighbour_scoring_impl(c) {
        panic!("neighbour_scoring benchmark setup failed: {err}");
    }
}

#[cfg(test)]
#[path = "benchmark_runner_tests.rs"]
mod tests;
