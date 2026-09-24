//! Tests for HNSW ef-sweep environment configuration.

#[rstest::rstest]
#[case::true_value(Some("true"), Some(true))]
#[case::false_value(Some("false"), Some(false))]
#[case::invalid_value(Some("unknown"), None)]
#[case::absent_value(None, None)]
fn parses_boolean_environment_values(#[case] value: Option<&str>, #[case] expected: Option<bool>) {
    let configured_value = value.map(str::to_owned);
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_RECALL_REPORT");
        configured_value.clone()
    });

    assert_eq!(
        super::environment::parse_bool_env_var(&env, "CHUTORO_BENCH_HNSW_RECALL_REPORT"),
        expected
    );
}

#[rstest::rstest]
#[case::recall_enabled(
    super::super::environment::should_collect_recall_report_with_env,
    "CHUTORO_BENCH_HNSW_RECALL_REPORT",
    "true"
)]
#[case::cluster_quality_enabled(
    super::super::environment::should_collect_cluster_quality_report_with_env,
    "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT",
    "true"
)]
fn explicit_report_enablement_overrides_benchmark_arguments(
    #[case] should_collect: fn(&dyn mockable::Env) -> bool,
    #[case] environment_key: &'static str,
    #[case] value: &'static str,
) {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(move |key| {
        assert_eq!(key, environment_key);
        Some(value.to_owned())
    });

    assert!(should_collect(&env));
}

#[rstest::rstest]
#[case::recall_disabled(
    super::super::environment::should_collect_recall_report_with_env,
    "CHUTORO_BENCH_HNSW_RECALL_REPORT"
)]
#[case::cluster_quality_disabled(
    super::super::environment::should_collect_cluster_quality_report_with_env,
    "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT"
)]
fn explicit_report_disablement_overrides_benchmark_arguments(
    #[case] should_collect: fn(&dyn mockable::Env) -> bool,
    #[case] environment_key: &'static str,
) {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(move |key| {
        assert_eq!(key, environment_key);
        Some("false".to_owned())
    });

    assert!(!should_collect(&env));
}

#[rstest::rstest]
#[case::recall(
    super::super::environment::should_collect_recall_report_with_env,
    "CHUTORO_BENCH_HNSW_RECALL_REPORT"
)]
#[case::cluster_quality(
    super::super::environment::should_collect_cluster_quality_report_with_env,
    "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT"
)]
fn absent_report_configuration_uses_discovery_mode(
    #[case] should_collect: fn(&dyn mockable::Env) -> bool,
    #[case] environment_key: &'static str,
) {
    let mut env = mockable::MockEnv::new();
    env.expect_string().returning(move |key| {
        assert_eq!(key, environment_key);
        None
    });

    assert_eq!(
        should_collect(&env),
        !super::environment::is_discovery_mode()
    );
}

#[test]
fn recall_report_path_defaults_when_environment_is_unset() {
    let mut env = mockable::MockEnv::new();
    env.expect_os_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_RECALL_REPORT_PATH");
        None
    });

    assert_eq!(
        super::environment::recall_report_path_with_env(&env),
        std::path::PathBuf::from(super::environment::RECALL_REPORT_PATH)
    );
}

#[test]
fn recall_report_path_uses_configured_os_string() {
    let configured_path = std::path::PathBuf::from("reports/recall.csv");
    let expected_path = configured_path.clone();
    let mut env = mockable::MockEnv::new();
    env.expect_os_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_RECALL_REPORT_PATH");
        Some(std::ffi::OsString::from(configured_path.clone()))
    });

    assert_eq!(
        super::environment::recall_report_path_with_env(&env),
        expected_path
    );
}

#[test]
fn cluster_quality_report_path_defaults_when_environment_is_unset() {
    let mut env = mockable::MockEnv::new();
    env.expect_os_string().returning(|key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH");
        None
    });

    assert_eq!(
        super::environment::cluster_quality_report_path_with_env(&env),
        std::path::PathBuf::from(super::environment::CLUSTERING_QUALITY_REPORT_PATH)
    );
}

#[test]
fn cluster_quality_report_path_uses_configured_os_string() {
    let configured_path = std::path::PathBuf::from("reports/cluster-quality.csv");
    let expected_path = configured_path.clone();
    let mut env = mockable::MockEnv::new();
    env.expect_os_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH");
        Some(std::ffi::OsString::from(configured_path.clone()))
    });

    assert_eq!(
        super::environment::cluster_quality_report_path_with_env(&env),
        expected_path
    );
}
