//! Tests for neighbour-scoring fixture construction and report orchestration.

#[expect(
    unused_imports,
    reason = "Criterion harness=false bench tests compile as ordinary code"
)]
use camino::Utf8Path;
#[expect(
    unused_imports,
    reason = "Criterion harness=false bench tests compile as ordinary code"
)]
use chutoro_benches::neighbour_scoring::{BUILD_PROFILE_REPORT, REPORT_DIR_NAME};
#[expect(
    unused_imports,
    reason = "Criterion harness=false bench tests compile as ordinary code"
)]
use chutoro_core::DataSource;
#[expect(
    unused_imports,
    reason = "Criterion harness=false bench tests compile as ordinary code"
)]
use tempfile::tempdir;

#[expect(
    unused_imports,
    reason = "Criterion harness=false bench tests compile as ordinary code"
)]
use super::{
    DEFAULT_BUILD_PROFILE_DIMENSION, DEFAULT_BUILD_PROFILE_POINT_COUNTS, LANE_REPORT, make_fixture,
    write_build_profile_report_for_point_counts, write_build_profile_report_with,
    write_lane_utilisation_report,
};

#[test]
fn fixture_contains_provider_rows_and_one_based_candidates() {
    let candidate_count = 8;
    let fixture = make_fixture(32, candidate_count).expect("fixture must be created");

    assert!(fixture.provider.len() >= candidate_count + 1);
    assert_eq!(
        fixture.candidates,
        (1..=candidate_count).collect::<Vec<_>>()
    );
}

#[test]
fn lane_utilisation_report_writes_expected_file() {
    let temp_dir = tempdir().expect("temp dir must be created");
    let report_parent_dir = Utf8Path::from_path(temp_dir.path()).expect("temp path must be UTF-8");

    let report_path = write_lane_utilisation_report(report_parent_dir)
        .expect("lane utilisation report must be written");

    assert_eq!(
        report_path,
        report_parent_dir.join(REPORT_DIR_NAME).join(LANE_REPORT),
    );
    assert!(report_path.exists());
}

#[test]
fn build_profile_report_respects_optional_target_and_writes_file() {
    let temp_dir = tempdir().expect("temp dir must be created");
    let report_parent_dir = Utf8Path::from_path(temp_dir.path()).expect("temp path must be UTF-8");
    let report_target = report_parent_dir
        .join(REPORT_DIR_NAME)
        .join(BUILD_PROFILE_REPORT);

    let skipped = write_build_profile_report_for_point_counts(None, &[16], 8)
        .expect("disabled build profile report must succeed");
    assert!(skipped.is_none());

    let written =
        write_build_profile_report_for_point_counts(Some(report_target.clone()), &[16], 8)
            .expect("enabled build profile report must succeed");

    assert_eq!(written.as_deref(), Some(report_target.as_path()));
    assert!(report_target.exists());
}

#[test]
fn build_profile_report_writes_to_supplied_target() {
    let temp_dir = tempdir().expect("temp dir must be created");
    let report_parent_dir = Utf8Path::from_path(temp_dir.path()).expect("temp path must be UTF-8");
    let report_target = report_parent_dir.join("custom-build-profile.csv");

    let written =
        write_build_profile_report_for_point_counts(Some(report_target.clone()), &[16], 8)
            .expect("build profile report must be written to the supplied target");

    assert_eq!(written.as_deref(), Some(report_target.as_path()));
    assert!(report_target.exists());
    assert!(
        !report_parent_dir
            .join(REPORT_DIR_NAME)
            .join(BUILD_PROFILE_REPORT)
            .exists()
    );
}

#[test]
fn default_build_profile_delegates_with_expected_configuration() {
    let report_target = Some(Utf8Path::new("target-parent").join("report.csv"));

    let result = write_build_profile_report_with(
        report_target.clone(),
        |actual_target, point_counts, dimension| {
            assert_eq!(actual_target, report_target);
            assert_eq!(point_counts, DEFAULT_BUILD_PROFILE_POINT_COUNTS);
            assert_eq!(dimension, DEFAULT_BUILD_PROFILE_DIMENSION);
            Ok(actual_target)
        },
    )
    .expect("default build profile delegation must succeed");

    assert_eq!(result, report_target);
}

#[test]
fn default_build_profile_configuration_is_stable() {
    assert_eq!(DEFAULT_BUILD_PROFILE_POINT_COUNTS, &[10_000, 100_000]);
    assert_eq!(DEFAULT_BUILD_PROFILE_DIMENSION, 128);
}
