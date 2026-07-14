//! Integration coverage for neighbour-scoring build-profile support seams.

use camino::Utf8Path;
use cap_std::{ambient_authority, fs_utf8::Dir};
use chutoro_benches::neighbour_scoring::{
    BUILD_PROFILE_REPORT, BuildProfileReportRow, LaneUtilisationReportRow, REPORT_DIR_NAME,
    build_profile_report_target_value, report_parent_dir_value, should_collect_build_profile_value,
    truthy_env_value, write_build_profile_report_csv, write_lane_utilisation_report_csv,
};
use rstest::{fixture, rstest};
use std::{io::Write, process::Command, time::Duration};
use tempfile::{NamedTempFile, TempDir, tempdir};
use thiserror::Error;

#[derive(Debug, Error)]
enum ReportFixtureError {
    #[error("report fixture I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("temp dir path is not UTF-8")]
    NonUtf8TempDir,
    #[error("expected report was not written: {0}")]
    ReportMissing(&'static str),
}

fn temp_dir_utf8_path(temp_dir: &tempfile::TempDir) -> Result<&Utf8Path, ReportFixtureError> {
    Utf8Path::from_path(temp_dir.path()).ok_or(ReportFixtureError::NonUtf8TempDir)
}

struct ReportDirectory {
    _temp_dir: TempDir,
    path: camino::Utf8PathBuf,
    root: Dir,
}

#[fixture]
fn report_directory() -> Result<ReportDirectory, ReportFixtureError> {
    let temp_dir = tempdir()?;
    let path = temp_dir_utf8_path(&temp_dir)?.to_path_buf();
    let root = Dir::open_ambient_dir(&path, ambient_authority())?;
    root.create_dir_all(REPORT_DIR_NAME)?;
    Ok(ReportDirectory {
        _temp_dir: temp_dir,
        path,
        root,
    })
}

#[rstest]
#[case::unset(None, false)]
#[case::empty(Some(""), false)]
#[case::false_word(Some("false"), false)]
#[case::zero(Some("0"), false)]
#[case::mixed_case_true(Some(" TrUe "), true)]
#[case::one(Some("1"), true)]
#[case::on(Some("on"), true)]
#[case::yes(Some("yes"), true)]
fn should_collect_build_profile_recognizes_env_values(
    #[case] value: Option<&str>,
    #[case] expected: bool,
) {
    assert_eq!(truthy_env_value(value), expected);
    assert_eq!(should_collect_build_profile_value(value), expected);
}

#[test]
fn report_path_uses_shared_report_directory_name() {
    let report_parent_dir = Utf8Path::new("/tmp/chutoro-target-dir");
    let actual_path = build_profile_report_target_value(Some("yes"), report_parent_dir);

    assert_eq!(REPORT_DIR_NAME, "benchmarks");
    assert_eq!(
        actual_path.as_deref(),
        Some(Utf8Path::new(
            "/tmp/chutoro-target-dir/benchmarks/neighbour_scoring_build_profile.csv",
        )),
    );
}

#[test]
fn neighbour_scoring_script_wires_expected_benchmark_binary() {
    let script = include_str!("../../scripts/bench-neighbour-scoring.sh");
    let mut script_file = NamedTempFile::new().expect("temp script file must be created");
    script_file
        .write_all(script.as_bytes())
        .expect("temp script file must be written");

    let shellcheck_output = Command::new("shellcheck")
        .arg("-s")
        .arg("bash")
        .arg(script_file.path())
        .output()
        .expect("shellcheck must run");

    assert!(script.contains("cargo bench -p chutoro-benches --bench neighbour_scoring --no-run"));
    assert!(script.contains(".target.name == \"neighbour_scoring\""));
    assert!(script.contains("${escaped_bench_binary} --bench --profile-time 1"));
    assert!(
        shellcheck_output.status.success(),
        "shellcheck failed: {}",
        String::from_utf8_lossy(&shellcheck_output.stderr),
    );
}

#[rstest]
fn lane_utilisation_report_file_generation_writes_schema_and_rows(
    #[from(report_directory)] report_directory_result: Result<ReportDirectory, ReportFixtureError>,
) -> Result<(), ReportFixtureError> {
    let report_directory = report_directory_result?;
    let mut file = report_directory
        .root
        .open_dir(REPORT_DIR_NAME)
        .and_then(|dir| dir.create("lane.csv"))
        .expect("report file must be created");

    write_lane_utilisation_report_csv(
        &mut file,
        [LaneUtilisationReportRow {
            bucket_kind: "realistic",
            candidate_count: 8,
        }],
    )
    .expect("lane utilisation report must be written");

    if !report_directory
        .path
        .join(REPORT_DIR_NAME)
        .join("lane.csv")
        .exists()
    {
        return Err(ReportFixtureError::ReportMissing("lane.csv"));
    }
    Ok(())
}

#[rstest]
fn build_profile_report_file_generation_writes_schema_and_rows(
    #[from(report_directory)] report_directory_result: Result<ReportDirectory, ReportFixtureError>,
) -> Result<(), ReportFixtureError> {
    let report_directory = report_directory_result?;
    let mut file = report_directory
        .root
        .open_dir(REPORT_DIR_NAME)
        .and_then(|dir| dir.create(BUILD_PROFILE_REPORT))
        .expect("report file must be created");

    write_build_profile_report_csv(
        &mut file,
        [BuildProfileReportRow {
            point_count: 16,
            dimension: 8,
            build_elapsed: Duration::from_millis(1),
            batch_scoring_time: Duration::from_micros(1),
            batch_calls: 1,
            scalar_calls: 0,
            total_batch_candidates: 8,
            min_batch: 8,
            max_batch: 8,
            median_batch: 8,
        }],
    )
    .expect("build profile report must be written");

    if !report_directory
        .path
        .join(REPORT_DIR_NAME)
        .join(BUILD_PROFILE_REPORT)
        .exists()
    {
        return Err(ReportFixtureError::ReportMissing(BUILD_PROFILE_REPORT));
    }
    Ok(())
}

#[rstest]
#[case::unset(None, false)]
#[case::empty(Some(""), false)]
#[case::false_word(Some("false"), false)]
#[case::truthy(Some("yes"), true)]
fn build_profile_report_target_tracks_env(#[case] value: Option<&str>, #[case] expected: bool) {
    let report_parent_dir = Utf8Path::new("/tmp/chutoro-target-dir");

    assert_eq!(should_collect_build_profile_value(value), expected);
    assert_eq!(
        build_profile_report_target_value(value, report_parent_dir).is_some(),
        expected,
    );
}

#[test]
fn build_profile_report_target_uses_expected_filename() {
    let expected_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../target/benchmarks/neighbour_scoring_build_profile.csv",
    );
    let report_parent_dir = report_parent_dir_value(None);
    let actual_path = build_profile_report_target_value(Some("yes"), &report_parent_dir);

    assert_eq!(actual_path.as_deref(), Some(Utf8Path::new(expected_path)));
}

fn assert_target_path_for_dir(report_parent_dir: &Utf8Path) {
    let actual_path = build_profile_report_target_value(Some("yes"), report_parent_dir);

    assert_eq!(
        actual_path.as_deref(),
        Some(Utf8Path::new(
            "/tmp/chutoro-target-dir/benchmarks/neighbour_scoring_build_profile.csv",
        )),
    );
}

#[test]
fn build_profile_report_target_reads_cargo_target_dir_env() {
    let report_parent_dir = report_parent_dir_value(Some("/tmp/chutoro-target-dir"));
    assert_target_path_for_dir(&report_parent_dir);
}

#[test]
fn build_profile_report_target_honours_cargo_target_dir() {
    let report_parent_dir = report_parent_dir_value(Some("/tmp/chutoro-target-dir"));
    assert_target_path_for_dir(&report_parent_dir);
}
