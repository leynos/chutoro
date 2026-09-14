//! Tests for neighbour-scoring benchmark orchestration.

use std::{cell::RefCell, io, rc::Rc};

use camino::Utf8PathBuf;
use criterion::{BenchmarkId, Criterion, Throughput};
use mockable::MockEnv;

use super::{
    BenchError, make_fixture, neighbour_scoring_impl_with, score_candidates, scoring_plan,
    should_use_short_measurement_value, should_use_short_measurement_with_env, throughput_for,
};
use crate::neighbour_scoring::all_buckets;

// `should_use_short_measurement_value` is a thin delegate to the canonical
// `truthy_env_value`, whose full truthy/falsy case table is exercised by
// `chutoro-benches/tests/neighbour_scoring_support.rs`. These two cases only
// confirm the delegation, not the whole table.
#[rstest::rstest]
#[case::falsy(Some("false"), false)]
#[case::truthy(Some("yes"), true)]
fn short_measurement_parser_delegates_to_truthy_env_value(
    #[case] value: Option<&str>,
    #[case] expected: bool,
) {
    assert_eq!(should_use_short_measurement_value(value), expected);
}

#[rstest::rstest]
#[case::unset(None, false)]
#[case::truthy(Some("true"), true)]
#[case::falsy(Some("false"), false)]
fn short_measurement_reads_environment(#[case] value: Option<&str>, #[case] expected: bool) {
    let configured_value = value.map(str::to_owned);
    let mut env = MockEnv::new();
    env.expect_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_NEIGHBOUR_SHORT_MEASUREMENT");
        configured_value.clone()
    });

    assert_eq!(should_use_short_measurement_with_env(&env), expected);
}

#[test]
fn score_candidates_returns_one_distance_per_candidate() {
    let candidate_count = 8;
    let fixture = make_fixture(32, candidate_count).expect("fixture must be created");
    let distances = score_candidates(&fixture, &fixture.candidates).expect("scoring must succeed");

    assert_eq!(distances.len(), candidate_count);
}

#[test]
fn throughput_conversion_uses_candidate_count() {
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
    let bucket = all_buckets()
        .next()
        .expect("neighbour scoring buckets must be non-empty");

    assert!(
        super::bench_id_for(bucket, 32) == BenchmarkId::new("realistic", "dim_32_candidates_8")
    );
}

#[test]
fn orchestration_writes_reports_before_all_scoring_cases() {
    let events = Rc::new(RefCell::new(Vec::new()));
    let lane_events = Rc::clone(&events);
    let build_events = Rc::clone(&events);
    let scoring_events = Rc::clone(&events);
    let mut criterion = Criterion::default();
    let env = configured_profile_env("false");

    neighbour_scoring_impl_with(
        &mut criterion,
        &env,
        (
            move |_| {
                lane_events.borrow_mut().push(("lane", 0, "", 0));
                Ok(())
            },
            move |_| {
                build_events.borrow_mut().push(("build", 0, "", 0));
                Ok(())
            },
        ),
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

    let mut expected_events = vec![("lane", 0, "", 0), ("build", 0, "", 0)];
    expected_events.extend(
        scoring_plan()
            .into_iter()
            .map(|(dimension, bucket)| ("score", dimension, bucket.kind_name(), bucket.size())),
    );

    assert_eq!(*events.borrow(), expected_events);
}

#[test]
fn orchestration_propagates_scoring_errors() {
    let mut criterion = Criterion::default();
    let env = configured_profile_env("false");

    let error =
        neighbour_scoring_impl_with(&mut criterion, &env, (|_| Ok(()), |_| Ok(())), |_, _, _| {
            Err(BenchError::Io(io::Error::other("scoring failed")))
        })
        .expect_err("scoring failure must propagate");

    assert!(matches!(error, BenchError::Io(_)));
}

#[test]
fn orchestration_omits_build_profile_directory_when_disabled() {
    assert_eq!(
        build_profile_directory("false")
            .expect("disabled build-profile configuration must succeed"),
        None
    );
}

#[test]
fn orchestration_passes_build_profile_directory_when_enabled() {
    let expected = Utf8PathBuf::from("configured-target");
    assert_eq!(
        build_profile_directory("true").expect("enabled build-profile configuration must succeed"),
        Some(expected)
    );
}

fn configured_profile_env(profile_value: &'static str) -> MockEnv {
    let mut env = MockEnv::new();
    env.expect_string().returning(move |key| match key {
        "CARGO_TARGET_DIR" => Some("configured-target".to_owned()),
        "CHUTORO_BENCH_NEIGHBOUR_PROFILE" => Some(profile_value.to_owned()),
        unexpected => panic!("unexpected environment key: {unexpected}"),
    });
    env
}

fn build_profile_directory(profile_value: &'static str) -> super::BenchResult<Option<Utf8PathBuf>> {
    let env = configured_profile_env(profile_value);
    let build_profile_directory = Rc::new(RefCell::new(None));
    let captured_directory = Rc::clone(&build_profile_directory);
    let mut criterion = Criterion::default();

    neighbour_scoring_impl_with(
        &mut criterion,
        &env,
        (
            |_| Ok(()),
            move |directory| {
                *captured_directory.borrow_mut() = directory.map(Utf8PathBuf::from);
                Ok(())
            },
        ),
        |_, _, _| Ok(()),
    )?;

    Ok(build_profile_directory.borrow().clone())
}
