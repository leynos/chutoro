//! Tests for HNSW ef-sweep environment configuration.

use camino::Utf8PathBuf;
use mockable::{Env, MockEnv};
use rstest::rstest;

use super::environment::{self, CLUSTERING_QUALITY_REPORT_PATH, RECALL_REPORT_PATH};

/// One report policy exposed by the shared environment module.
#[derive(Clone, Copy)]
struct ReportPolicy {
    should_collect: fn(&dyn Env, &[&str]) -> bool,
    should_collect_with_env: fn(&dyn Env) -> bool,
    path_with_env: fn(&dyn Env) -> Utf8PathBuf,
    enabled_key: &'static str,
    path_key: &'static str,
    default_path: &'static str,
}

/// The recall and clustering-quality report policies under test.
const REPORT_POLICIES: [ReportPolicy; 2] = [
    ReportPolicy {
        should_collect: environment::should_collect_recall_report_with_args,
        should_collect_with_env: environment::should_collect_recall_report_with_env,
        path_with_env: environment::recall_report_path_with_env,
        enabled_key: "CHUTORO_BENCH_HNSW_RECALL_REPORT",
        path_key: "CHUTORO_BENCH_HNSW_RECALL_REPORT_PATH",
        default_path: RECALL_REPORT_PATH,
    },
    ReportPolicy {
        should_collect: environment::should_collect_cluster_quality_report_with_args,
        should_collect_with_env: environment::should_collect_cluster_quality_report_with_env,
        path_with_env: environment::cluster_quality_report_path_with_env,
        enabled_key: "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT",
        path_key: "CHUTORO_BENCH_HNSW_CLUSTER_QUALITY_REPORT_PATH",
        default_path: CLUSTERING_QUALITY_REPORT_PATH,
    },
];

#[rstest]
#[case::true_value(Some("true"), Some(true))]
#[case::false_value(Some("false"), Some(false))]
#[case::invalid_value(Some("unknown"), None)]
#[case::absent_value(None, None)]
fn parses_boolean_environment_values(#[case] value: Option<&str>, #[case] expected: Option<bool>) {
    let configured_value = value.map(str::to_owned);
    let mut env = MockEnv::new();
    env.expect_string().returning(move |key| {
        assert_eq!(key, "CHUTORO_BENCH_HNSW_RECALL_REPORT");
        configured_value.clone()
    });

    let env_var_name = environment::EnvVarName::new("CHUTORO_BENCH_HNSW_RECALL_REPORT");
    assert_eq!(
        environment::parse_bool_env_var(&env, env_var_name),
        expected
    );
}

#[rstest]
#[case::enabled_overrides_list(Some("true"), vec!["bench", "--list"], true)]
#[case::disabled_overrides_normal(Some("false"), vec!["bench"], false)]
#[case::normal_default(None, vec!["bench"], true)]
#[case::list_default(None, vec!["bench", "--list"], false)]
#[case::exact_default(None, vec!["bench", "--exact"], false)]
fn report_collection_follows_environment_then_explicit_arguments(
    #[case] configured: Option<&str>,
    #[case] args: Vec<&str>,
    #[case] expected: bool,
) {
    for policy in REPORT_POLICIES {
        let mut env = MockEnv::new();
        let configured_value = configured.map(str::to_owned);
        let enabled_key = policy.enabled_key;
        env.expect_string().returning(move |key| {
            assert_eq!(key, enabled_key);
            configured_value.clone()
        });

        assert_eq!(
            (policy.should_collect)(&env, &args),
            expected,
            "collection policy for {}",
            policy.enabled_key
        );
    }
}

/// The explicit environment override must win over process arguments.
///
/// This exercises the process-argument wrapper used by the benchmark
/// binaries. The assertion is independent of Criterion's arguments because
/// the environment value alone decides the outcome.
#[test]
fn explicit_enablement_overrides_process_arguments() {
    for policy in REPORT_POLICIES {
        let mut env = MockEnv::new();
        let enabled_key = policy.enabled_key;
        env.expect_string().returning(move |key| {
            assert_eq!(key, enabled_key);
            Some("true".to_owned())
        });

        assert!(
            (policy.should_collect_with_env)(&env),
            "process-argument wrapper for {}",
            policy.enabled_key
        );
    }
}

#[rstest]
#[case::unset_defaults(None)]
#[case::configured_os_string(Some("reports/custom.csv"))]
fn report_paths_follow_environment_configuration(#[case] configured: Option<&str>) {
    for policy in REPORT_POLICIES {
        let mut env = MockEnv::new();
        let configured_value = configured.map(std::ffi::OsString::from);
        let path_key = policy.path_key;
        env.expect_os_string().returning(move |key| {
            assert_eq!(key, path_key);
            configured_value.clone()
        });

        let expected =
            configured.map_or_else(|| Utf8PathBuf::from(policy.default_path), Utf8PathBuf::from);

        assert_eq!(
            (policy.path_with_env)(&env),
            expected,
            "path policy for {}",
            policy.path_key
        );
    }
}
