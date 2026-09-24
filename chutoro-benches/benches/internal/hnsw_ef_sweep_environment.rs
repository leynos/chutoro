//! Environment policy shared by the HNSW ef-sweep benchmark and its tests.

use std::path::PathBuf;

use mockable::Env;

/// Report destination for recall-versus-ef_construction metrics.
pub(super) const RECALL_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_recall_vs_ef.csv"
);

/// Report destination for ARI/NMI-versus-ef_construction metrics.
pub(super) const CLUSTERING_QUALITY_REPORT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../target/benchmarks/hnsw_cluster_quality_vs_ef.csv"
);

/// Emits a warning to stderr for an unrecognized env var value.
#[expect(
    clippy::print_stderr,
    reason = "Benchmark-only diagnostic for invalid env var; no structured logging available."
)]
fn warn_unrecognized_bool_env(env_var_name: &str, value: &str) {
    eprintln!(
        "warning: unrecognized value {value:?} for \
         {env_var_name}; expected 0/1/true/false/on/off"
    );
}

/// Parse one optional benchmark boolean environment variable.
pub(super) fn parse_bool_env_var(env: &dyn Env, env_var_name: &str) -> Option<bool> {
    let value = env.string(env_var_name)?;
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "0" | "false" | "off") {
        return Some(false);
    }
    if matches!(normalized.as_str(), "1" | "true" | "on") {
        return Some(true);
    }
    warn_unrecognized_bool_env(env_var_name, &value);
    None
}

/// Determine whether Criterion is enumerating benchmark names.
pub(super) fn is_discovery_mode() -> bool {
    std::env::args().any(|arg| arg == "--list" || arg == "--exact")
}

/// Read recall-report configuration through an injected environment reader.
pub(super) fn should_collect_recall_report_with_env(env: &dyn Env) -> bool {
    parse_bool_env_var(env, "CHUTORO_BENCH_HNSW_RECALL_REPORT")
        .unwrap_or_else(|| !is_discovery_mode())
}

/// Resolve the recall-report path through an injected environment reader.
pub(super) fn recall_report_path_with_env(env: &dyn Env) -> PathBuf {
    env.os_string("CHUTORO_BENCH_HNSW_RECALL_REPORT_PATH")
        .map_or_else(|| PathBuf::from(RECALL_REPORT_PATH), PathBuf::from)
}

/// Read clustering-quality configuration through an injected reader.
pub(super) fn should_collect_cluster_quality_report_with_env(env: &dyn Env) -> bool {
    parse_bool_env_var(env, "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT")
        .unwrap_or_else(|| !is_discovery_mode())
}

/// Resolve the clustering-quality path through an injected reader.
pub(super) fn cluster_quality_report_path_with_env(env: &dyn Env) -> PathBuf {
    env.os_string("CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH")
        .map_or_else(
            || PathBuf::from(CLUSTERING_QUALITY_REPORT_PATH),
            PathBuf::from,
        )
}
