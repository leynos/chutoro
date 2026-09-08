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
    let has_injected_coverage_marker =
        env.os_string("CARGO_LLVM_COV").is_some() || env.os_string("LLVM_PROFILE_FILE").is_some();

    cfg!(coverage)
        || option_env!("CARGO_LLVM_COV").is_some()
        || option_env!("LLVM_PROFILE_FILE").is_some()
        || has_injected_coverage_marker
}

#[cfg(test)]
mod tests {
    //! Tests for injected coverage-job configuration.

    use mockable::MockEnv;

    use super::is_coverage_job_with_env;

    /// Return whether this build already carries a compile-time coverage marker.
    fn has_compile_time_coverage_marker() -> bool {
        cfg!(coverage)
            || option_env!("CARGO_LLVM_COV").is_some()
            || option_env!("LLVM_PROFILE_FILE").is_some()
    }

    #[test]
    fn cargo_llvm_cov_marker_enables_coverage_job() {
        let mut env = MockEnv::new();
        env.expect_os_string().returning(|key| {
            assert_eq!(key, "CARGO_LLVM_COV");
            Some("1".into())
        });

        assert!(is_coverage_job_with_env(&env));
    }

    #[test]
    fn llvm_profile_file_marker_enables_coverage_job() {
        let mut env = MockEnv::new();
        env.expect_os_string().returning(|key| match key {
            "CARGO_LLVM_COV" => None,
            "LLVM_PROFILE_FILE" => Some("default_%p.profraw".into()),
            unexpected => panic!("unexpected environment key: {unexpected}"),
        });

        assert!(is_coverage_job_with_env(&env));
    }

    #[test]
    fn empty_environment_is_not_a_coverage_job_without_compile_time_marker() {
        if has_compile_time_coverage_marker() {
            return;
        }

        let mut env = MockEnv::new();
        env.expect_os_string().returning(|key| match key {
            "CARGO_LLVM_COV" | "LLVM_PROFILE_FILE" => None,
            unexpected => panic!("unexpected environment key: {unexpected}"),
        });

        assert!(!is_coverage_job_with_env(&env));
    }
}
