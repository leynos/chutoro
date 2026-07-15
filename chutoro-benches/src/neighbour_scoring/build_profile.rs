//! Build-profile report path and environment helpers.

use camino::{Utf8Path, Utf8PathBuf};

/// Environment variable that enables HNSW build-profile report generation.
pub const BUILD_PROFILE_ENV: &str = "CHUTORO_BENCH_NEIGHBOUR_PROFILE";

/// Build-profile report filename.
pub const BUILD_PROFILE_REPORT: &str = "neighbour_scoring_build_profile.csv";

const CARGO_TARGET_DIR_ENV: &str = "CARGO_TARGET_DIR";
const DEFAULT_REPORT_PARENT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../target");
/// Directory below the report parent where benchmark diagnostics are written.
pub const REPORT_DIR_NAME: &str = "benchmarks";

/// A benchmark report location expressed as its configured parent and filename.
///
/// Keeping these components together prevents report writers from bypassing
/// the shared [`REPORT_DIR_NAME`] convention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReportTarget {
    report_parent_dir: Utf8PathBuf,
    filename: String,
}

impl ReportTarget {
    /// Returns the configured Cargo target directory containing benchmark reports.
    #[must_use]
    pub fn report_parent_dir(&self) -> &Utf8Path {
        &self.report_parent_dir
    }

    /// Returns the report filename created below [`REPORT_DIR_NAME`].
    #[must_use]
    pub fn filename(&self) -> &str {
        &self.filename
    }

    /// Returns the complete conventional path for this report.
    #[must_use]
    pub fn path(&self) -> Utf8PathBuf {
        self.report_parent_dir
            .join(REPORT_DIR_NAME)
            .join(&self.filename)
    }
}

/// Returns whether an environment value is a truthy benchmark flag.
///
/// # Examples
///
/// ```
/// use chutoro_benches::neighbour_scoring::truthy_env_value;
///
/// assert!(truthy_env_value(Some(" yes ")));
/// assert!(!truthy_env_value(Some("false")));
/// ```
#[must_use]
pub fn truthy_env_value(value: Option<&str>) -> bool {
    value.is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

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
    truthy_env_value(value)
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
/// use chutoro_benches::neighbour_scoring::report_parent_dir_value;
///
/// assert_eq!(report_parent_dir_value(Some("target")).as_str(), "target");
/// ```
#[must_use]
pub fn report_parent_dir_value(cargo_target_dir: Option<&str>) -> Utf8PathBuf {
    cargo_target_dir.map_or_else(
        || Utf8PathBuf::from(DEFAULT_REPORT_PARENT_DIR),
        Utf8PathBuf::from,
    )
}

/// Reads the process environment and returns the parent directory used for
/// benchmark reports.
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
    report_parent_dir_value(std::env::var(CARGO_TARGET_DIR_ENV).ok().as_deref())
}

/// Returns a benchmark report target below the supplied report parent directory.
///
/// # Examples
///
/// ```
/// use camino::Utf8Path;
/// use chutoro_benches::neighbour_scoring::{
///     BUILD_PROFILE_REPORT, report_path_value,
/// };
///
/// assert_eq!(
///     report_path_value(Utf8Path::new("target"), BUILD_PROFILE_REPORT)
///         .path()
///         .file_name(),
///     Some(BUILD_PROFILE_REPORT)
/// );
/// ```
#[must_use]
pub fn report_path_value(report_parent_dir: &Utf8Path, filename: &str) -> ReportTarget {
    ReportTarget {
        report_parent_dir: report_parent_dir.to_path_buf(),
        filename: filename.to_owned(),
    }
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
pub fn report_path(filename: &str) -> ReportTarget {
    report_path_value(&report_parent_dir(), filename)
}

/// Returns the build-profile report target when collection is enabled.
///
/// # Examples
///
/// ```
/// use camino::Utf8Path;
/// use chutoro_benches::neighbour_scoring::build_profile_report_target_value;
///
/// let report_parent_dir = Utf8Path::new("target");
/// assert!(build_profile_report_target_value(Some("true"), report_parent_dir).is_some());
/// assert!(build_profile_report_target_value(Some("false"), report_parent_dir).is_none());
/// ```
#[must_use]
pub fn build_profile_report_target_value(
    value: Option<&str>,
    report_parent_dir: &Utf8Path,
) -> Option<ReportTarget> {
    should_collect_build_profile_value(value)
        .then(|| report_path_value(report_parent_dir, BUILD_PROFILE_REPORT))
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
pub fn build_profile_report_target() -> Option<ReportTarget> {
    let report_parent_dir = report_parent_dir();
    build_profile_report_target_value(
        std::env::var(BUILD_PROFILE_ENV).ok().as_deref(),
        &report_parent_dir,
    )
}
