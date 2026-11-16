//! Environment policy shared by the HNSW ef-sweep benchmark and its tests.

use camino::Utf8PathBuf;
use mockable::Env;

#[path = "environment_support.rs"]
mod environment_support;
pub(super) use environment_support::EnvVarName;

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
fn warn_unrecognized_bool_env(env_var_name: EnvVarName<'_>, value: &str) {
    eprintln!(
        concat!(
            "warning: unrecognized value {:?} for ",
            "{}; expected 0/1/true/false/on/off"
        ),
        value,
        env_var_name.as_str()
    );
}

/// Parse one optional benchmark boolean environment variable.
pub(super) fn parse_bool_env_var(env: &dyn Env, env_var_name: EnvVarName<'_>) -> Option<bool> {
    let value = env.string(env_var_name.as_str())?;
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

/// Read recall-report configuration through an injected environment reader.
pub(super) fn should_collect_recall_report_with_env(env: &dyn Env) -> bool {
    let args: Vec<String> = std::env::args().collect();
    let argument_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    should_collect_recall_report_with_args(env, &argument_refs)
}

/// Read the recall-report decision from explicit benchmark arguments.
pub(super) fn should_collect_recall_report_with_args(env: &dyn Env, args: &[&str]) -> bool {
    let enabled = EnvVarName::new("CHUTORO_BENCH_HNSW_RECALL_REPORT");
    parse_bool_env_var(env, enabled).unwrap_or_else(|| !is_discovery_mode_args(args))
}

/// Resolve the recall-report path through an injected environment reader.
pub(super) fn recall_report_path_with_env(env: &dyn Env) -> Utf8PathBuf {
    let path_key = EnvVarName::new("CHUTORO_BENCH_HNSW_RECALL_REPORT_PATH");
    configured_report_path(env, path_key).unwrap_or_else(|| Utf8PathBuf::from(RECALL_REPORT_PATH))
}

/// Read clustering-quality configuration through an injected reader.
pub(super) fn should_collect_cluster_quality_report_with_env(env: &dyn Env) -> bool {
    let args: Vec<String> = std::env::args().collect();
    let argument_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    should_collect_cluster_quality_report_with_args(env, &argument_refs)
}

/// Read the clustering-quality decision from explicit benchmark arguments.
pub(super) fn should_collect_cluster_quality_report_with_args(
    env: &dyn Env,
    args: &[&str],
) -> bool {
    let enabled = EnvVarName::new("CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT");
    parse_bool_env_var(env, enabled).unwrap_or_else(|| !is_discovery_mode_args(args))
}

/// Resolve the clustering-quality path through an injected reader.
pub(super) fn cluster_quality_report_path_with_env(env: &dyn Env) -> Utf8PathBuf {
    let path_key = EnvVarName::new("CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH");
    configured_report_path(env, path_key)
        .unwrap_or_else(|| Utf8PathBuf::from(CLUSTERING_QUALITY_REPORT_PATH))
}

/// Determine whether explicit arguments describe Criterion discovery.
fn is_discovery_mode_args(args: &[&str]) -> bool {
    args.iter().any(|arg| matches!(*arg, "--list" | "--exact"))
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
