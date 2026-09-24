//! Shared environment policy for the HNSW benchmark and its focused tests.

use std::path::PathBuf;

use chutoro_benches::criterion_support::{is_benchmark_discovery, is_exact_benchmark_probe};
use mockable::Env;

/// Report destination for derived memory metrics.
pub(super) const MEMORY_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_memory_profile.csv"
);

/// Determine whether this invocation should collect HNSW memory measurements.
pub(super) fn should_collect_memory_profile_with_env(env: &dyn Env) -> bool {
    if let Some(value) = env.string("CHUTORO_BENCH_HNSW_MEMORY_PROFILE") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "0" | "false" | "off") {
            return false;
        }
        if matches!(normalized.as_str(), "1" | "true" | "on") {
            return true;
        }
    }
    !is_benchmark_discovery() && !is_exact_benchmark_probe()
}

/// Resolve the memory-report path through an injected environment reader.
pub(super) fn memory_report_path_with_env(env: &dyn Env) -> PathBuf {
    env.os_string("CHUTORO_BENCH_HNSW_MEMORY_REPORT_PATH")
        .map_or_else(|| PathBuf::from(MEMORY_REPORT_PATH), PathBuf::from)
}

/// Determine whether the diverse-source benchmark should include MNIST.
pub(super) fn should_include_mnist_with_env(env: &dyn Env) -> bool {
    env.string("CHUTORO_BENCH_ENABLE_MNIST").as_deref() == Some("1")
}
