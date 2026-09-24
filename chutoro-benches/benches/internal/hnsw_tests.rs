//! Tests for HNSW memory-profile environment configuration.

use super::environment::*;
use chutoro_benches::criterion_support::{is_benchmark_discovery, is_exact_benchmark_probe};

#[rstest::fixture]
fn env_with_memory_settings_unset() -> mockable::MockEnv {
    let mut env = mockable::MockEnv::new();

    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_PROFILE");
        None
    });
    env.expect_os_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_REPORT_PATH");
        None
    });

    env
}

#[test]
fn memory_profile_explicitly_disabled_by_environment() {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_PROFILE");
        Some("false".to_owned())
    });

    assert!(!should_collect_memory_profile_with_env(&env));
}

#[test]
fn memory_profile_explicitly_enabled_by_environment() {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_PROFILE");
        Some("true".to_owned())
    });

    assert!(should_collect_memory_profile_with_env(&env));
}

#[test]
fn memory_profile_uses_benchmark_mode_when_environment_is_unset() {
    let env = env_with_memory_settings_unset();

    assert_eq!(
        should_collect_memory_profile_with_env(&env),
        !is_benchmark_discovery() && !is_exact_benchmark_probe()
    );
}

#[test]
fn memory_report_path_defaults_when_environment_is_unset() {
    let env = env_with_memory_settings_unset();

    assert_eq!(
        memory_report_path_with_env(&env),
        std::path::PathBuf::from(MEMORY_REPORT_PATH)
    );
}

#[test]
fn memory_report_path_uses_configured_os_string() {
    let configured_path = std::path::PathBuf::from("reports/hnsw-memory.csv");
    let expected_path = configured_path.clone();
    let mut env = mockable::MockEnv::new();
    env.expect_os_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_REPORT_PATH");
        Some(std::ffi::OsString::from(configured_path.clone()))
    });

    assert_eq!(memory_report_path_with_env(&env), expected_path);
}

#[test]
fn mnist_is_excluded_when_environment_is_unset() {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_ENABLE_MNIST");
        None
    });

    assert!(!should_include_mnist_with_env(&env));
}

#[test]
fn mnist_is_excluded_when_environment_is_disabled() {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_ENABLE_MNIST");
        Some("0".to_owned())
    });

    assert!(!should_include_mnist_with_env(&env));
}

#[test]
fn mnist_is_included_when_environment_is_enabled() {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_ENABLE_MNIST");
        Some("1".to_owned())
    });

    assert!(should_include_mnist_with_env(&env));
}
