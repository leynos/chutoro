//! Integration coverage for neighbour-scoring build-profile support seams.

use chutoro_benches::neighbour_scoring::{
    BUILD_PROFILE_ENV, BUILD_PROFILE_REPORT, build_profile_report_target, report_path,
    should_collect_build_profile, should_collect_build_profile_value,
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
    let _profile_env = EnvVarGuard::set(BUILD_PROFILE_ENV, "yes");
    let expected_path = report_path(BUILD_PROFILE_REPORT);
    let actual_path = build_profile_report_target();

    assert_eq!(actual_path, Some(expected_path));
}
