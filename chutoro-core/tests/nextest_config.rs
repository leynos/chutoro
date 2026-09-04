//! Regression checks for the repository's nextest profile configuration.

use rstest::rstest;

const NEXTEST_CONFIG: &str = include_str!("../../.config/nextest.toml");
const PROPERTY_TESTS_WORKFLOW: &str = include_str!("../../.github/workflows/property-tests.yml");
const BENCHMARK_REGRESSIONS_WORKFLOW: &str =
    include_str!("../../.github/workflows/benchmark-regressions.yml");
const MAKEFILE: &str = include_str!("../../Makefile");
const BENCH_SLOW_TIMEOUT: &str =
    "slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }";

const LONG_BENCH_SLOW_TIMEOUT: &str =
    "slow-timeout = { period = \"900s\", terminate-after = 1, grace-period = \"5s\" }";
const TRYBUILD_SLOW_TIMEOUT: &str =
    "slow-timeout = { period = \"300s\", terminate-after = 1, grace-period = \"5s\" }";
const NESTED_BENCH_SMOKE_TIMEOUT: &str = TRYBUILD_SLOW_TIMEOUT;

fn override_blocks(profile_name: &str) -> Vec<&'static str> {
    NEXTEST_CONFIG
        .split(&format!("[[profile.{profile_name}.overrides]]"))
        .skip(1)
        .map(str::trim)
        .collect()
}

/// Reports whether `profile_name` declares an override block containing every
/// fragment in `fragments`.
fn has_override_with_fragments(profile_name: &str, fragments: &[&str]) -> bool {
    override_blocks(profile_name)
        .into_iter()
        .any(|block| fragments.iter().all(|fragment| block.contains(fragment)))
}

/// Asserts that a profile declares an override block matching every fragment.
///
/// A macro so a failure reports the calling test's line rather than a shared
/// helper's line.
macro_rules! assert_override_present {
    ($profile:expr, [$($fragment:expr),+ $(,)?] $(,)?) => {
        assert!(
            has_override_with_fragments($profile, &[$($fragment),+]),
            "profile '{}' should declare an override containing {:?}",
            $profile,
            [$($fragment),+],
        );
    };
}

fn extract_block(
    haystack: &'static str,
    header: &str,
    terminator: &str,
    label: &str,
) -> Result<&'static str, String> {
    let (_, rest) = haystack
        .split_once(header)
        .ok_or_else(|| format!("{label} not found"))?;
    let block = match rest.split_once(terminator) {
        Some((block, _)) => block,
        None => rest,
    };

    Ok(block)
}

fn workflow_job_block(job: &str) -> Result<&'static str, String> {
    extract_block(
        PROPERTY_TESTS_WORKFLOW,
        &format!("  {job}:"),
        "\n\n  ",
        &format!("workflow job '{job}'"),
    )
}

fn benchmark_workflow_job_block(job: &str) -> Result<&'static str, String> {
    extract_block(
        BENCHMARK_REGRESSIONS_WORKFLOW,
        &format!("  {job}:"),
        "\n\n  ",
        &format!("benchmark workflow job '{job}'"),
    )
}

fn make_target_block(target: &str) -> Result<&'static str, String> {
    extract_block(
        MAKEFILE,
        &format!("\n{target}:"),
        "\n\n",
        &format!("Makefile target '{target}'"),
    )
}

#[test]
fn nextest_default_profile_keeps_global_timeout_guard() {
    assert!(NEXTEST_CONFIG.contains("global-timeout = \"40m\""));
}

#[rstest]
#[case(
    "filter = \"package(chutoro-benches) & kind(bench)\"",
    LONG_BENCH_SLOW_TIMEOUT
)]
#[case(
    "filter = \"package(chutoro-benches) & test(/extract_labels\\\\//)\"",
    BENCH_SLOW_TIMEOUT
)]
#[case(
    "filter = \"package(chutoro-benches) & test(/edge_harvest_construction\\\\//)\"",
    BENCH_SLOW_TIMEOUT
)]
fn nextest_default_profile_keeps_benchmark_timeout_guards(
    #[case] filter_value: &str,
    #[case] expected_timeout: &str,
) {
    assert_override_present!(
        "default",
        [filter_value, "threads-required = 8", expected_timeout]
    );
}

#[test]
fn property_tests_pr_timeout_covers_hnsw_idempotency_budget() {
    assert_override_present!(
        "ci",
        [
            "filter = \"test(/hnsw_idempotency_preserved_proptest/)\"",
            BENCH_SLOW_TIMEOUT,
        ]
    );

    let pr_job = workflow_job_block("property-tests-pr").expect("property-tests-pr job must exist");
    assert!(pr_job.contains("timeout-minutes: 20"));
}

#[test]
fn default_profile_covers_idempotency_rstest_case_4_timeout() {
    assert_override_present!(
        "default",
        [
            "filter = \"test(/idempotency_rstest_cases::case_4/)\"",
            "period = \"180s\"",
        ]
    );
}

#[rstest]
#[case("default")]
#[case("ci")]
fn nextest_profiles_keep_trybuild_timeout_guards(#[case] profile_name: &str) {
    assert_override_present!(
        profile_name,
        [
            "portable_simd_gating_compile_checks",
            "session_api_compiles_when_cpu_feature_is_enabled",
            "arrow_parquet_types_share_one_family",
            "threads-required = 4",
            TRYBUILD_SLOW_TIMEOUT,
        ]
    );
}

#[test]
fn default_profile_serializes_nested_benchmark_smoke_test() {
    assert_override_present!(
        "default",
        [
            "benchmark_binaries_cover_discovery_and_exact_smoke_paths",
            "threads-required = 8",
            NESTED_BENCH_SMOKE_TIMEOUT,
        ]
    );
}

#[rstest]
#[case("default")]
#[case("ci")]
fn profiles_preserve_write_lock_proptest_timeout(#[case] profile_name: &str) {
    assert_override_present!(
        profile_name,
        [
            "generated_hnsw_scoring_does_not_run_inside_write_graph_scope",
            BENCH_SLOW_TIMEOUT,
        ]
    );
}

#[test]
fn makefile_exposes_typecheck_gate() {
    let typecheck_block = make_target_block("typecheck").expect("typecheck target must exist");
    assert!(MAKEFILE.contains(" typecheck "));
    assert!(typecheck_block.contains("cargo") || typecheck_block.contains("$(CARGO)"));
    assert!(typecheck_block.contains("check --workspace --all-targets --all-features"));
    assert!(typecheck_block.contains("$(BUILD_JOBS)"));
}

#[test]
fn benchmark_smoke_job_covers_hnsw_exact_probe() {
    let smoke_job =
        benchmark_workflow_job_block("benchmark-smoke").expect("benchmark-smoke job must exist");

    assert!(
        smoke_job
            .contains("cargo bench -p chutoro-benches --bench \"${{ matrix.bench }}\" -- --list")
    );
    assert!(smoke_job.contains("if: ${{ matrix.bench == 'hnsw' }}"));
    assert!(smoke_job.contains("cargo bench -p chutoro-benches --bench hnsw --"));
    assert!(smoke_job.contains("--exact"));
    assert!(smoke_job.contains("/tmp/bench-smoke-${{ matrix.bench }}.log"));
    assert!(
        smoke_job
            .contains("${{ matrix.bench == 'hnsw' && '/tmp/bench-smoke-hnsw-exact.log' || '' }}")
    );
}

/// The `ci` profile runs four test threads, which is why the shape each
/// property job runs on is worth pinning rather than leaving to whoever
/// edits the workflow next.
///
/// The pull-request job ran on eight cores until the shape was measured.
/// Its whole "Run property suite" step, compilation and 250 cases together,
/// took 21 to 24 seconds across the four suites on eight cores (run
/// 33852441511), inside jobs of 51 to 57 seconds, four ways in parallel and
/// off the critical path. Eight cores bought nothing worth paying for, so
/// it runs on two and the profile oversubscribes them. That doubles the step,
/// to 52 to 55 seconds, and grows the whole job from 51 to 57 seconds to 79
/// to 87 seconds: a deliberate trade of about thirty seconds of parallel
/// wall time for roughly 60 % of this job's cost. The full before and after
/// table is in the developers guide. The weekly job is off the feedback path
/// and runs on GitHub's four-core hosted runner.
///
/// Runner placement policy, tool installation, and cache ownership are
/// asserted in `tests/workflow_contracts/`; this test covers only the
/// relationship between the profile and the shapes.
#[rstest]
#[case("property-tests-pr", "runs-on: ubicloud-standard-2")]
#[case("property-tests-weekly", "runs-on: ubuntu-latest")]
fn property_tests_runners_satisfy_the_ci_thread_count(
    #[case] job: &str,
    #[case] expected_runner: &str,
) {
    assert!(NEXTEST_CONFIG.contains("[profile.ci]"));
    assert!(NEXTEST_CONFIG.contains("test-threads = 4"));

    let job_block = workflow_job_block(job).expect("property test job must exist");
    assert!(
        job_block.contains(expected_runner),
        "{job} should declare '{expected_runner}'",
    );
}
