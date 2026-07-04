//! Shared Criterion benchmark helpers.
//!
//! Provides small utilities for benchmark binaries that need to expose cheap
//! discovery or exact-probe paths without running expensive setup work.

use std::fmt::Display;

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};

/// Returns whether the current command line includes `flag`.
#[must_use]
pub fn is_cli_flag_present(flag: &str) -> bool {
    std::env::args().any(|arg| arg == flag)
}

/// Registers no-op benchmark functions for each parameter value.
///
/// This is used by probe-only paths such as `--list` and selected `--exact`
/// smoke tests, where Criterion must see stable benchmark identifiers but the
/// real benchmark setup would be too expensive for discovery.
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
