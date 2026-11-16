//! Shared environment policy for the HNSW benchmark and its focused tests.

use camino::Utf8PathBuf;
use chutoro_benches::criterion_support::{
    is_benchmark_discovery_args, is_exact_benchmark_probe_args,
};
use mockable::Env;

#[path = "environment_support.rs"]
mod environment_support;
pub(super) use environment_support::EnvVarName;

/// Report destination for derived memory metrics.
pub(super) const MEMORY_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_memory_profile.csv"
);

/// Determine whether this invocation should collect HNSW memory measurements.
pub(super) fn should_collect_memory_profile_with_env(env: &dyn Env) -> bool {
    let args: Vec<String> = std::env::args().collect();
    let argument_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    should_collect_memory_profile_with_args(env, &argument_refs)
}

/// Determine the collection decision from explicit benchmark arguments.
///
/// Taking the arguments as a parameter keeps the benchmark-mode defaults
/// testable without deriving an expectation from the detectors under test.
pub(super) fn should_collect_memory_profile_with_args(env: &dyn Env, args: &[&str]) -> bool {
    if let Some(value) = env.string("CHUTORO_BENCH_HNSW_MEMORY_PROFILE") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "0" | "false" | "off") {
            return false;
        }
        if matches!(normalized.as_str(), "1" | "true" | "on") {
            return true;
        }
    }
    !is_benchmark_discovery_args(args) && !is_exact_benchmark_probe_args(args)
}

/// Resolve the memory-report path through an injected environment reader.
pub(super) fn memory_report_path_with_env(env: &dyn Env) -> Utf8PathBuf {
    configured_report_path(
        env,
        EnvVarName::new("CHUTORO_BENCH_HNSW_MEMORY_REPORT_PATH"),
    )
    .unwrap_or_else(|| Utf8PathBuf::from(MEMORY_REPORT_PATH))
}

/// Determine whether the diverse-source benchmark should include MNIST.
pub(super) fn should_include_mnist_with_env(env: &dyn Env) -> bool {
    env.string("CHUTORO_BENCH_ENABLE_MNIST").as_deref() == Some("1")
}

/// Read a configured report path, rejecting values that are not UTF-8.
fn configured_report_path(env: &dyn Env, env_var_name: EnvVarName<'_>) -> Option<Utf8PathBuf> {
    let configured = env.os_string(env_var_name.as_str())?;
    Utf8PathBuf::from_os_string(configured).map_or_else(
        |_| {
            warn_non_utf8_path(env_var_name);
            None
        },
        Some,
    )
}

/// Emits a warning to stderr when a configured path is not valid UTF-8.
#[expect(
    clippy::print_stderr,
    reason = "Benchmark-only diagnostic for an unusable report path; no structured logging available."
)]
fn warn_non_utf8_path(env_var_name: EnvVarName<'_>) {
    eprintln!(
        "warning: ignoring non-UTF-8 path in {}",
        env_var_name.as_str()
    );
}
