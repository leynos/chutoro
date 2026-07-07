//! Shared Criterion benchmark helpers.
//!
//! Provides small utilities for benchmark binaries that need cheap discovery
//! or exact-probe paths without running expensive setup work.

use std::{fmt::Display, time::Duration};

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};

/// Returns whether the current command line includes `flag`.
///
/// ```no_run
/// # use chutoro_benches::criterion_support::is_cli_flag_present;
/// if is_cli_flag_present("--list") {
///     // Skip expensive setup when Criterion is only listing benchmarks.
/// }
/// ```
#[must_use]
pub fn is_cli_flag_present(flag: &str) -> bool {
    args_contain_flag(std::env::args(), flag)
}

fn args_contain_flag(args: impl IntoIterator<Item = impl AsRef<str>>, flag: &str) -> bool {
    args.into_iter().any(|arg| arg.as_ref() == flag)
}

/// Returns whether Criterion is listing benchmark identifiers.
///
/// ```no_run
/// # use chutoro_benches::criterion_support::is_benchmark_discovery;
/// if is_benchmark_discovery() {
///     // Register placeholder benchmarks without expensive setup.
/// }
/// ```
#[must_use]
pub fn is_benchmark_discovery() -> bool {
    is_cli_flag_present("--list")
}

/// Returns whether Criterion is probing one exact benchmark case.
///
/// # Examples
///
/// ```
/// use chutoro_benches::criterion_support::is_exact_benchmark_probe_args;
///
/// assert!(is_exact_benchmark_probe_args(["bench", "--exact"]));
/// assert!(!is_exact_benchmark_probe_args(["bench", "--list"]));
/// ```
#[must_use]
pub fn is_exact_benchmark_probe_args<I, S>(args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    args_contain_flag(args, "--exact")
}

/// Reads process arguments and returns whether Criterion is probing one exact
/// benchmark case.
///
/// # Examples
///
/// ```
/// use chutoro_benches::criterion_support::is_exact_benchmark_probe;
///
/// let _is_exact = is_exact_benchmark_probe();
/// ```
#[must_use]
pub fn is_exact_benchmark_probe() -> bool {
    is_exact_benchmark_probe_args(std::env::args())
}

/// Returns whether nextest is running one Criterion exact benchmark case.
///
/// ```no_run
/// # use chutoro_benches::criterion_support::is_nextest_exact_benchmark_probe;
/// if is_nextest_exact_benchmark_probe() {
///     // Register bounded probe cases without running expensive setup.
/// }
/// ```
#[must_use]
pub fn is_nextest_exact_benchmark_probe() -> bool {
    is_nextest_exact_benchmark_probe_args(
        std::env::args(),
        std::env::var_os("NEXTEST_TEST_NAME").is_some(),
    )
}

fn is_nextest_exact_benchmark_probe_args<I, S>(args: I, has_nextest_test_name: bool) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    is_exact_benchmark_probe_args(args) && has_nextest_test_name
}

/// Returns whether an exact Criterion probe should use a labelled short path.
///
/// # Examples
///
/// ```
/// use chutoro_benches::criterion_support::should_short_circuit_exact_label_probe_args;
///
/// assert!(should_short_circuit_exact_label_probe_args(
///     ["bench", "--exact"],
///     "text",
///     "text",
/// ));
/// assert!(!should_short_circuit_exact_label_probe_args(
///     ["bench", "--list"],
///     "text",
///     "text",
/// ));
/// ```
#[must_use]
pub fn should_short_circuit_exact_label_probe_args<I, S>(
    args: I,
    bench_label: &str,
    exact_probe_label: &str,
) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    is_exact_benchmark_probe_args(args) && bench_label == exact_probe_label
}

/// Registers no-op benchmark functions for each parameter value.
///
/// This is used by probe-only paths such as `--list` and selected `--exact`
/// smoke tests, where Criterion must see stable benchmark identifiers but the
/// real benchmark setup would be too expensive for discovery.
///
/// ```no_run
/// # use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};
/// # use chutoro_benches::criterion_support::register_noop_benches;
/// # fn configure_group(_: &mut BenchmarkGroup<'_, WallTime>) {}
/// let mut criterion = Criterion::default();
/// let params = [100usize, 500, 1_000];
/// register_noop_benches(&mut criterion, "parallel_kruskal", params, configure_group);
/// ```
pub fn register_noop_benches<P>(
    c: &mut Criterion,
    group_name: &str,
    params: impl IntoIterator<Item = P>,
    configure_group: impl FnOnce(&mut BenchmarkGroup<'_, WallTime>),
) where
    P: Display,
{
    let mut group = c.benchmark_group(group_name);
    configure_group(&mut group);

    for param in params {
        group.bench_function(BenchmarkId::from_parameter(&param), |b| {
            b.iter(|| ());
        });
    }

    group.finish();
}

/// Applies common short-run timing to a Criterion group.
///
/// # Examples
///
/// ```no_run
/// use criterion::Criterion;
/// use chutoro_benches::criterion_support::configure_short_measurement_group;
///
/// let mut criterion = Criterion::default();
/// let mut group = criterion.benchmark_group("example");
/// configure_short_measurement_group(&mut group, 10, true);
/// ```
pub fn configure_short_measurement_group(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sample_size: usize,
    should_use_short_measurement: bool,
) {
    group.sample_size(sample_size);
    if should_use_short_measurement {
        group.warm_up_time(Duration::from_millis(1));
        group.measurement_time(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for Criterion command-line helper behaviour.

    use rstest::rstest;

    use super::{
        args_contain_flag, is_exact_benchmark_probe_args, is_nextest_exact_benchmark_probe_args,
        should_short_circuit_exact_label_probe_args,
    };

    #[test]
    fn args_contain_flag_detects_present_flag() {
        let args = ["bench", "--list", "hnsw_build"];
        assert!(args_contain_flag(args, "--list"));
    }

    #[test]
    fn args_contain_flag_rejects_absent_flag() {
        let args = ["bench", "--exact", "hnsw_build"];
        assert!(!args_contain_flag(args, "--list"));
    }

    #[rstest]
    #[case::exact(["bench", "--exact"], true)]
    #[case::list(["bench", "--list"], false)]
    #[case::embedded(["bench", "not--exact"], false)]
    fn exact_probe_detection_reads_literal_argument(
        #[case] args: [&str; 2],
        #[case] expected: bool,
    ) {
        assert_eq!(is_exact_benchmark_probe_args(args), expected);
    }

    #[rstest]
    #[case::nextest_exact(["bench", "--exact"], true, true)]
    #[case::ordinary_exact(["bench", "--exact"], false, false)]
    #[case::nextest_list(["bench", "--list"], true, false)]
    fn nextest_exact_probe_requires_exact_arg_and_nextest_marker(
        #[case] args: [&str; 2],
        #[case] has_nextest_test_name: bool,
        #[case] expected: bool,
    ) {
        assert_eq!(
            is_nextest_exact_benchmark_probe_args(args, has_nextest_test_name),
            expected,
        );
    }

    #[rstest]
    #[case::exact_label(["bench", "--exact"], "text_levenshtein", true)]
    #[case::discovery_label(["bench", "--list"], "text_levenshtein", false)]
    #[case::exact_other_label(["bench", "--exact"], "gaussian_blobs", false)]
    fn exact_label_probe_short_circuit_requires_exact_arg_and_matching_label(
        #[case] args: [&str; 2],
        #[case] bench_label: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            should_short_circuit_exact_label_probe_args(args, bench_label, "text_levenshtein",),
            expected,
        );
    }
}
