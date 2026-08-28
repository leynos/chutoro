//! Shared helpers for CPU HNSW tests.

use mockable::{DefaultEnv, Env};

/// Detects whether the current test run is coverage-instrumented.
///
/// Coverage builds can perturb scheduling and substantially increase the cost
/// of some property and parallel-construction tests.
pub(super) fn is_coverage_job() -> bool {
    is_coverage_job_with_env(&DefaultEnv)
}

/// Detect coverage configuration through an injected environment reader.
fn is_coverage_job_with_env(env: &dyn Env) -> bool {
    cfg!(coverage)
        || option_env!("CARGO_LLVM_COV").is_some()
        || option_env!("LLVM_PROFILE_FILE").is_some()
        || env.os_string("CARGO_LLVM_COV").is_some()
        || env.os_string("LLVM_PROFILE_FILE").is_some()
}
