//! Neighbour scoring benchmarks for HNSW candidate evaluation.
//!
//! The harness measures the query-centric dense-provider path at realistic
//! HNSW candidate bucket sizes and emits small diagnostic CSV reports under
//! `target/benchmarks/` so roadmap item 2.3.1 can be closed on evidence.

use camino::Utf8Path;
use chutoro_benches::{
    criterion_support::configure_short_measurement_group,
    neighbour_scoring::{
        BUILD_PROFILE_ENV, ReportTarget, build_profile_report_target_value, report_parent_dir,
        truthy_env_value,
    },
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
    configure_short_measurement_group(group, 10, should_use_short_measurement());
}

fn should_use_short_measurement_value(value: Option<&str>) -> bool {
    truthy_env_value(value)
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

fn throughput_for(bucket: CandidateBucket) -> BenchResult<Throughput> {
    let throughput =
        u64::try_from(bucket.size()).map_err(|source| BenchError::CandidateCountConversion {
            candidate_count: bucket.size(),
            source,
        })?;
    Ok(Throughput::Elements(throughput))
}

fn bench_id_for(bucket: CandidateBucket, dimension: usize) -> BenchmarkId {
    BenchmarkId::new(
        bucket.kind_name(),
        format!("dim_{dimension}_candidates_{}", bucket.size()),
    )
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
    group.throughput(throughput_for(bucket)?);
    let id = bench_id_for(bucket, dimension);
    group.bench_with_input(id, &fixture, |b, scoring_fixture| {
        b.iter(|| run_scoring_iteration(scoring_fixture, &scoring_fixture.candidates));
    });
    Ok(())
}

fn scoring_plan() -> Vec<(usize, CandidateBucket)> {
    DIMENSIONS
        .iter()
        .flat_map(|&dimension| all_buckets().map(move |bucket| (dimension, bucket)))
        .collect()
}

fn neighbour_scoring_impl(c: &mut Criterion) -> BenchResult<()> {
    neighbour_scoring_impl_with(
        c,
        |report_parent_dir| write_lane_utilisation_report(report_parent_dir).map(drop),
        |report_parent_dir| write_build_profile_report(report_parent_dir).map(drop),
        bench_case,
    )
}

fn neighbour_scoring_impl_with(
    c: &mut Criterion,
    lane_report_writer: impl FnOnce(&Utf8Path) -> BenchResult<()>,
    build_profile_writer: impl FnOnce(Option<&Utf8Path>) -> BenchResult<()>,
    mut scoring_case: impl FnMut(
        &mut BenchmarkGroup<'_, WallTime>,
        usize,
        CandidateBucket,
    ) -> BenchResult<()>,
) -> BenchResult<()> {
    let report_parent_dir = report_parent_dir();
    let build_profile_target = build_profile_report_target_value(
        std::env::var(BUILD_PROFILE_ENV).ok().as_deref(),
        &report_parent_dir,
    );
    let build_profile_report_dir = build_profile_target
        .as_ref()
        .map(ReportTarget::report_parent_dir);
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

#[cfg(test)]
mod tests {
    //! Tests for neighbour-scoring benchmark orchestration.

    #[rstest::rstest]
    #[case::unset(None, false)]
    #[case::empty(Some(""), false)]
    #[case::false_word(Some("false"), false)]
    #[case::zero(Some("0"), false)]
    #[case::mixed_case_true(Some(" TrUe "), true)]
    #[case::one(Some("1"), true)]
    #[case::on(Some("on"), true)]
    #[case::yes(Some("yes"), true)]
    fn short_measurement_parser_recognizes_env_values(
        #[case] value: Option<&str>,
        #[case] expected: bool,
    ) {
        use super::should_use_short_measurement_value;

        assert_eq!(should_use_short_measurement_value(value), expected);
    }

    #[test]
    fn score_candidates_returns_one_distance_per_candidate() {
        use super::{score_candidates, support::make_fixture};

        let candidate_count = 8;
        let fixture = make_fixture(32, candidate_count).expect("fixture must be created");
        let distances =
            score_candidates(&fixture, &fixture.candidates).expect("scoring must succeed");

        assert_eq!(distances.len(), candidate_count);
    }

    #[test]
    fn throughput_conversion_uses_candidate_count() {
        use super::{support::all_buckets, throughput_for};
        use criterion::Throughput;

        let bucket = all_buckets()
            .next()
            .expect("neighbour scoring buckets must be non-empty");

        assert!(matches!(
            throughput_for(bucket).expect("throughput conversion must succeed"),
            Throughput::Elements(8),
        ));
    }

    #[test]
    fn benchmark_id_uses_kind_dimension_and_candidate_count() {
        use super::{bench_id_for, support::CandidateBucket};

        let bucket = CandidateBucket::realistic_for_test(8);

        assert_eq!(
            bench_id_for(bucket, 32).to_string(),
            "realistic/dim_32_candidates_8",
        );
    }

    #[test]
    fn scoring_plan_covers_each_dimension_and_bucket_once() {
        use super::{DIMENSIONS, scoring_plan, support::all_buckets};

        let plan = scoring_plan();
        let buckets = all_buckets().collect::<Vec<_>>();

        assert_eq!(plan.len(), DIMENSIONS.len() * buckets.len());
        for &dimension in DIMENSIONS {
            for bucket in &buckets {
                let occurrences = plan
                    .iter()
                    .filter(|(planned_dimension, planned_bucket)| {
                        *planned_dimension == dimension
                            && planned_bucket.kind_name() == bucket.kind_name()
                            && planned_bucket.size() == bucket.size()
                    })
                    .count();
                assert_eq!(occurrences, 1);
            }
        }
    }

    #[test]
    fn orchestration_writes_reports_before_all_scoring_cases() {
        use std::{cell::RefCell, rc::Rc};

        use super::{neighbour_scoring_impl_with, scoring_plan};
        use criterion::Criterion;

        let events = Rc::new(RefCell::new(Vec::new()));
        let lane_events = Rc::clone(&events);
        let build_events = Rc::clone(&events);
        let scoring_events = Rc::clone(&events);
        let mut criterion = Criterion::default();

        neighbour_scoring_impl_with(
            &mut criterion,
            move |_| {
                lane_events.borrow_mut().push(("lane", 0, "", 0));
                Ok(())
            },
            move |_| {
                build_events.borrow_mut().push(("build", 0, "", 0));
                Ok(())
            },
            move |_, dimension, bucket| {
                scoring_events.borrow_mut().push((
                    "score",
                    dimension,
                    bucket.kind_name(),
                    bucket.size(),
                ));
                Ok(())
            },
        )
        .expect("orchestration must succeed");

        let events = events.borrow();
        assert_eq!(events[0], ("lane", 0, "", 0));
        assert_eq!(events[1], ("build", 0, "", 0));
        let expected_cases = scoring_plan()
            .into_iter()
            .map(|(dimension, bucket)| ("score", dimension, bucket.kind_name(), bucket.size()))
            .collect::<Vec<_>>();
        assert_eq!(events[2..], expected_cases);
    }

    #[test]
    fn orchestration_propagates_scoring_errors() {
        use std::io;

        use super::{BenchError, neighbour_scoring_impl_with};
        use criterion::Criterion;

        let mut criterion = Criterion::default();

        let error = neighbour_scoring_impl_with(
            &mut criterion,
            |_| Ok(()),
            |_| Ok(()),
            |_, _, _| Err(BenchError::Io(io::Error::other("scoring failed"))),
        )
        .expect_err("scoring failure must propagate");

        assert!(matches!(error, BenchError::Io(_)));
    }
}
