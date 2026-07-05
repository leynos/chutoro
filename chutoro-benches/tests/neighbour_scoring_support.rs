//! Integration coverage for neighbour-scoring build-profile support seams.

use camino::Utf8Path;
use chutoro_benches::neighbour_scoring::{
    BUILD_PROFILE_ENV, build_profile_report_target, build_profile_report_target_value,
    report_parent_dir, report_parent_dir_value, should_collect_build_profile,
    should_collect_build_profile_value,
};
use chutoro_test_support::env::EnvVarGuard;
use rstest::rstest;

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
    assert_eq!(should_collect_build_profile_value(value), expected);
}

#[rstest]
#[case::unset(None, false)]
#[case::empty(Some(""), false)]
#[case::false_word(Some("false"), false)]
#[case::truthy(Some("yes"), true)]
fn build_profile_report_target_tracks_env(#[case] value: Option<&str>, #[case] expected: bool) {
    let _profile_env = value.map_or_else(
        || EnvVarGuard::remove(BUILD_PROFILE_ENV),
        |raw| EnvVarGuard::set(BUILD_PROFILE_ENV, raw),
    );

    assert_eq!(should_collect_build_profile(), expected);
    assert_eq!(build_profile_report_target().is_some(), expected);
}

#[test]
fn build_profile_report_target_uses_expected_filename() {
    let _target_dir = EnvVarGuard::remove("CARGO_TARGET_DIR");
    let expected_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../target/benchmarks/neighbour_scoring_build_profile.csv",
    );
    let report_parent_dir = report_parent_dir();
    let actual_path = build_profile_report_target_value(Some("yes"), &report_parent_dir);

    assert_eq!(actual_path.as_deref(), Some(Utf8Path::new(expected_path)));
}

#[test]
fn build_profile_report_target_reads_cargo_target_dir_env() {
    let _target_dir = EnvVarGuard::set("CARGO_TARGET_DIR", "/tmp/chutoro-target-dir");
    let report_parent_dir = report_parent_dir();
    let actual_path = build_profile_report_target_value(Some("yes"), &report_parent_dir);

    assert_eq!(
        actual_path.as_deref(),
        Some(Utf8Path::new(
            "/tmp/chutoro-target-dir/benchmarks/neighbour_scoring_build_profile.csv",
        )),
    );
}

#[test]
fn build_profile_report_target_honours_cargo_target_dir() {
    let report_parent_dir = report_parent_dir_value(Some("/tmp/chutoro-target-dir"));
    let actual_path = build_profile_report_target_value(Some("yes"), &report_parent_dir);

    assert_eq!(
        actual_path.as_deref(),
        Some(Utf8Path::new(
            "/tmp/chutoro-target-dir/benchmarks/neighbour_scoring_build_profile.csv",
        )),
    );
}
