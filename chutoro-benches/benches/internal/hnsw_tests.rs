//! Tests for HNSW memory-profile environment configuration.

use camino::Utf8PathBuf;
use mockable::MockEnv;
use rstest::rstest;

use super::environment::*;

#[rstest]
#[case::enabled_overrides_list(Some("true"), vec!["bench", "--list"], true)]
#[case::disabled_overrides_normal(Some("false"), vec!["bench"], false)]
#[case::normal_default(None, vec!["bench"], true)]
#[case::list_default(None, vec!["bench", "--list"], false)]
#[case::exact_default(None, vec!["bench", "--exact"], false)]
fn memory_profile_follows_environment_then_explicit_arguments(
    #[case] configured: Option<&str>,
    #[case] args: Vec<&str>,
    #[case] expected: bool,
) {
    let mut env = MockEnv::new();
    let configured_value = configured.map(str::to_owned);
    env.expect_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_PROFILE");
        configured_value.clone()
    });

    assert_eq!(
        should_collect_memory_profile_with_args(&env, &args),
        expected
    );
}

/// The explicit environment override must win over process arguments.
///
/// Exercises the process-argument wrapper the benchmark binary calls. The
/// environment value alone decides the outcome, so the assertion does not
/// depend on Criterion's own arguments.
#[test]
fn memory_profile_environment_enablement_overrides_process_arguments() {
    let mut env = MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_PROFILE");
        Some("true".to_owned())
    });

    assert!(should_collect_memory_profile_with_env(&env));
}

#[rstest]
#[case::unset_defaults(None, MEMORY_REPORT_PATH)]
#[case::configured_os_string(Some("reports/hnsw-memory.csv"), "reports/hnsw-memory.csv")]
fn memory_report_path_follows_environment_configuration(
    #[case] configured: Option<&str>,
    #[case] expected: &str,
) {
    let mut env = MockEnv::new();
    let configured_value = configured.map(std::ffi::OsString::from);
    env.expect_os_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_MEMORY_REPORT_PATH");
        configured_value.clone()
    });

    assert_eq!(
        memory_report_path_with_env(&env),
        Utf8PathBuf::from(expected)
    );
}

#[test]
fn mnist_is_excluded_when_environment_is_unset() {
    let mut env = MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_ENABLE_MNIST");
        None
    });

    assert!(!should_include_mnist_with_env(&env));
}

#[test]
fn mnist_is_excluded_when_environment_is_disabled() {
    let mut env = MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_ENABLE_MNIST");
        Some("0".to_owned())
    });

    assert!(!should_include_mnist_with_env(&env));
}

#[test]
fn mnist_is_included_when_environment_is_enabled() {
    let mut env = MockEnv::new();
    env.expect_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_ENABLE_MNIST");
        Some("1".to_owned())
    });

    assert!(should_include_mnist_with_env(&env));
}
