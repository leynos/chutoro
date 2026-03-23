//! Shared helpers for CPU HNSW tests.

/// Detects whether the current test run is coverage-instrumented.
///
/// Coverage builds can perturb scheduling and substantially increase the cost
/// of some property and parallel-construction tests.
pub(super) fn is_coverage_job() -> bool {
    cfg!(coverage)
        || option_env!("CARGO_LLVM_COV").is_some()
        || option_env!("LLVM_PROFILE_FILE").is_some()
        || std::env::var_os("CARGO_LLVM_COV").is_some()
        || std::env::var_os("LLVM_PROFILE_FILE").is_some()
}
