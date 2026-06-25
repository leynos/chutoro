//! Build-profile report path and environment helpers.

use camino::Utf8PathBuf;

/// Environment variable that enables HNSW build-profile report generation.
pub const BUILD_PROFILE_ENV: &str = "CHUTORO_BENCH_NEIGHBOUR_PROFILE";

/// Build-profile report filename.
pub const BUILD_PROFILE_REPORT: &str = "neighbour_scoring_build_profile.csv";

const CARGO_TARGET_DIR_ENV: &str = "CARGO_TARGET_DIR";
const DEFAULT_REPORT_PARENT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../target");
const REPORT_DIR_NAME: &str = "benchmarks";

/// Returns whether optional build-profile collection is enabled.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::should_collect_build_profile_value;
///
/// assert!(should_collect_build_profile_value(Some("yes")));
/// assert!(!should_collect_build_profile_value(Some("0")));
/// ```
#[must_use]
pub fn should_collect_build_profile_value(value: Option<&str>) -> bool {
    value.is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

/// Reads the process environment and returns whether build-profile collection
/// is enabled.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::should_collect_build_profile;
///
/// let _enabled = should_collect_build_profile();
/// ```
#[must_use]
pub fn should_collect_build_profile() -> bool {
    should_collect_build_profile_value(std::env::var(BUILD_PROFILE_ENV).ok().as_deref())
}

/// Returns the parent directory used for benchmark reports.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::report_parent_dir;
///
/// assert!(!report_parent_dir().as_str().is_empty());
/// ```
#[must_use]
pub fn report_parent_dir() -> Utf8PathBuf {
    std::env::var(CARGO_TARGET_DIR_ENV).map_or_else(
        |_| Utf8PathBuf::from(DEFAULT_REPORT_PARENT_DIR),
        Utf8PathBuf::from,
    )
}

/// Returns the full path for a benchmark report filename.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::{BUILD_PROFILE_REPORT, report_path};
///
/// assert_eq!(
///     report_path(BUILD_PROFILE_REPORT).file_name(),
///     Some(BUILD_PROFILE_REPORT)
/// );
/// ```
#[must_use]
pub fn report_path(filename: &str) -> Utf8PathBuf {
    report_parent_dir().join(REPORT_DIR_NAME).join(filename)
}

/// Returns the build-profile report target when collection is enabled.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::build_profile_report_target_value;
///
/// assert!(build_profile_report_target_value(Some("true")).is_some());
/// assert!(build_profile_report_target_value(Some("false")).is_none());
/// ```
#[must_use]
pub fn build_profile_report_target_value(value: Option<&str>) -> Option<Utf8PathBuf> {
    should_collect_build_profile_value(value).then(|| report_path(BUILD_PROFILE_REPORT))
}

/// Reads the process environment and returns the build-profile report target
/// when collection is enabled.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::build_profile_report_target;
///
/// let _target = build_profile_report_target();
/// ```
#[must_use]
pub fn build_profile_report_target() -> Option<Utf8PathBuf> {
    build_profile_report_target_value(std::env::var(BUILD_PROFILE_ENV).ok().as_deref())
}
