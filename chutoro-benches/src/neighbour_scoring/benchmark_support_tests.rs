//! Tests for neighbour-scoring fixture construction and report orchestration.

#[test]
fn fixture_contains_provider_rows_and_one_based_candidates() {
    use chutoro_core::DataSource;

    use super::make_fixture;

    let candidate_count = 8;
    let fixture = make_fixture(32, candidate_count).expect("fixture must be created");

    assert!(fixture.provider.len() > candidate_count);
    assert_eq!(
        fixture.candidates,
        (1..=candidate_count).collect::<Vec<_>>()
    );
}

#[test]
fn lane_utilisation_report_writes_expected_file() {
    use crate::neighbour_scoring::REPORT_DIR_NAME;
    use camino::Utf8Path;
    use tempfile::tempdir;

    use super::{LANE_REPORT, write_lane_utilisation_report};

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
fn build_profile_report_writes_conventional_file() {
    use crate::neighbour_scoring::{BUILD_PROFILE_REPORT, REPORT_DIR_NAME, report_path_value};
    use camino::Utf8Path;
    use tempfile::tempdir;

    use super::write_build_profile_report_for_point_counts;

    let temp_dir = tempdir().expect("temp dir must be created");
    let report_parent_dir = Utf8Path::from_path(temp_dir.path()).expect("temp path must be UTF-8");

    let report_target = report_path_value(report_parent_dir, BUILD_PROFILE_REPORT);
    let written = write_build_profile_report_for_point_counts(report_target, &[16], 8)
        .expect("enabled build profile report must succeed");

    assert_eq!(
        written.path(),
        report_parent_dir
            .join(REPORT_DIR_NAME)
            .join(BUILD_PROFILE_REPORT),
    );
    assert!(written.path().exists());
}

#[test]
fn default_build_profile_delegates_with_expected_configuration() {
    use crate::neighbour_scoring::{BUILD_PROFILE_REPORT, report_path_value};
    use camino::Utf8Path;

    use super::{
        DEFAULT_BUILD_PROFILE_DIMENSION, DEFAULT_BUILD_PROFILE_POINT_COUNTS,
        write_build_profile_report_with,
    };

    let report_parent_dir = Utf8Path::new("target-parent");
    let expected_target = report_path_value(report_parent_dir, BUILD_PROFILE_REPORT);

    let result = write_build_profile_report_with(
        Some(report_parent_dir),
        |actual_parent_dir, point_counts, dimension| {
            assert_eq!(actual_parent_dir, report_parent_dir);
            assert_eq!(point_counts, DEFAULT_BUILD_PROFILE_POINT_COUNTS);
            assert_eq!(dimension, DEFAULT_BUILD_PROFILE_DIMENSION);
            Ok(report_path_value(actual_parent_dir, BUILD_PROFILE_REPORT))
        },
    )
    .expect("default build profile delegation must succeed");

    assert_eq!(result, Some(expected_target));
}

#[test]
fn build_profile_report_with_skips_writer_without_parent_directory() {
    use super::write_build_profile_report_with;

    let result = write_build_profile_report_with(None, |_, _, _| {
        panic!("writer must not run when build profiling is disabled");
    })
    .expect("disabled build profile report must succeed");

    assert_eq!(result, None);
}

#[test]
fn build_profile_report_honours_custom_target_filename() {
    use crate::neighbour_scoring::{REPORT_DIR_NAME, report_path_value};
    use camino::Utf8Path;
    use cap_std::{ambient_authority, fs_utf8::Dir};
    use tempfile::tempdir;

    use super::write_build_profile_report_for_point_counts;

    let temp_dir = tempdir().expect("temp dir must be created");
    let report_parent_dir = Utf8Path::from_path(temp_dir.path()).expect("temp path must be UTF-8");
    let report_target = report_path_value(report_parent_dir, "custom-build-profile.csv");

    let written = write_build_profile_report_for_point_counts(report_target.clone(), &[16], 8)
        .expect("custom build profile report must succeed");
    let report_dir = Dir::open_ambient_dir(report_parent_dir, ambient_authority())
        .expect("report parent directory must be opened")
        .open_dir(REPORT_DIR_NAME)
        .expect("report directory must be opened");
    let contents = report_dir
        .read_to_string(written.filename())
        .expect("custom build profile report must be readable");

    assert_eq!(written, report_target);
    assert!(contents.starts_with(concat!(
        "point_count,dimension,build_seconds,accumulated_batch_scoring_seconds,",
        "accumulated_batch_scoring_vs_wall_basis_points,batch_calls,scalar_calls,",
        "total_batch_candidates,min_batch,max_batch,median_batch\n",
        "16,8,",
    )));
}

#[test]
fn default_build_profile_configuration_is_stable() {
    use super::{DEFAULT_BUILD_PROFILE_DIMENSION, DEFAULT_BUILD_PROFILE_POINT_COUNTS};

    assert_eq!(DEFAULT_BUILD_PROFILE_POINT_COUNTS, &[10_000, 100_000]);
    assert_eq!(DEFAULT_BUILD_PROFILE_DIMENSION, 128);
}
