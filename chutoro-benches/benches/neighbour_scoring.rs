//! Neighbour scoring benchmarks for HNSW candidate evaluation.
//!
//! The harness measures the query-centric dense-provider path at realistic
//! HNSW candidate bucket sizes and emits small diagnostic CSV reports under
//! `target/benchmarks/` so roadmap item 2.3.1 can be closed on evidence.

use std::time::Duration;

use chutoro_benches::neighbour_scoring::{
    BUILD_PROFILE_ENV, build_profile_report_target_value, report_parent_dir,
};
use chutoro_core::DataSource;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, black_box, criterion_main,
    measurement::WallTime,
};
#[path = "neighbour_scoring/profiling.rs"]
mod profiling;
#[path = "neighbour_scoring/support.rs"]
mod support;

use support::{
    BenchError, BenchResult, CandidateBucket, DIMENSIONS, ScoringFixture, all_buckets,
    make_fixture, write_build_profile_report, write_lane_utilisation_report,
};

const QUERY_INDEX: usize = 0;
const SHORT_MEASUREMENT_ENV: &str = "CHUTORO_BENCH_NEIGHBOUR_SHORT_MEASUREMENT";

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    if should_use_short_measurement() {
        group.warm_up_time(Duration::from_millis(1));
        group.measurement_time(Duration::from_millis(10));
    }
}

fn should_use_short_measurement_value(value: Option<&str>) -> bool {
    value.is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

fn should_use_short_measurement() -> bool {
    should_use_short_measurement_value(std::env::var(SHORT_MEASUREMENT_ENV).ok().as_deref())
}

fn score_candidates(
    scoring_fixture: &ScoringFixture,
    candidates: &[usize],
) -> Result<Vec<f32>, chutoro_core::DataSourceError> {
    scoring_fixture
        .provider
        .batch_distances(black_box(QUERY_INDEX), black_box(candidates))
}

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

fn bench_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    bucket: CandidateBucket,
) -> BenchResult<()> {
    let fixture = make_fixture(dimension, bucket.size())?;
    score_candidates(&fixture, &fixture.candidates)?;
    let throughput =
        u64::try_from(bucket.size()).map_err(|source| BenchError::CandidateCountConversion {
            candidate_count: bucket.size(),
            source,
        })?;
    group.throughput(Throughput::Elements(throughput));
    let id = BenchmarkId::new(
        bucket.kind_name(),
        format!("dim_{dimension}_candidates_{}", bucket.size()),
    );
    group.bench_with_input(id, &fixture, |b, scoring_fixture| {
        b.iter(|| run_scoring_iteration(scoring_fixture, &scoring_fixture.candidates));
    });
    Ok(())
}

fn neighbour_scoring_impl(c: &mut Criterion) -> BenchResult<()> {
    let report_parent_dir = report_parent_dir();
    let build_profile_target = build_profile_report_target_value(
        std::env::var(BUILD_PROFILE_ENV).ok().as_deref(),
        &report_parent_dir,
    );
    let _lane_report = write_lane_utilisation_report(&report_parent_dir)?;
    let _build_profile = write_build_profile_report(&report_parent_dir, build_profile_target)?;
    let mut group = c.benchmark_group("neighbour_scoring");
    configure_group(&mut group);
    for &dimension in DIMENSIONS {
        for bucket in all_buckets() {
            bench_case(&mut group, dimension, bucket)?;
        }
    }
    group.finish();
    Ok(())
}

fn neighbour_scoring(c: &mut Criterion) {
    if let Err(err) = neighbour_scoring_impl(c) {
        panic!("neighbour_scoring benchmark setup failed: {err}");
    }
}

mod bench_harness {
    //! Criterion benchmark entrypoint for neighbour-scoring measurements.

    use super::neighbour_scoring;
    use criterion::criterion_group;

    criterion_group!(benches, neighbour_scoring);
}

criterion_main!(bench_harness::benches);
