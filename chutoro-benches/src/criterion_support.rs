//! Shared Criterion benchmark helpers.
//!
//! Provides small utilities for benchmark binaries that need to expose cheap
//! discovery or exact-probe paths without running expensive setup work.

use std::fmt::Display;

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

#[cfg(test)]
mod tests {
    use super::args_contain_flag;

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
}
