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
            "arrow_parquet_types_share_one_family",
            "portable_simd_gating_compile_checks",
            "session_api_compiles_when_cpu_feature_is_enabled",
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

#[rstest]
#[case("property-tests-pr")]
#[case("property-tests-weekly")]
fn property_tests_use_eight_core_runner(#[case] job: &str) {
    let job_block = workflow_job_block(job).expect("property test job must exist");
    assert!(job_block.contains("runs-on: ubicloud-standard-8"));
}

#[rstest]
#[case("property-tests-pr")]
#[case("property-tests-weekly")]
fn property_tests_install_locked_cached_nextest(#[case] job: &str) {
    let job_block = workflow_job_block(job).expect("property test job must exist");
    assert!(PROPERTY_TESTS_WORKFLOW.contains("CARGO_NEXTEST_VERSION: \"0.9.143\""));
    assert!(job_block.contains("cache_nextest_step"));
    assert!(PROPERTY_TESTS_WORKFLOW.contains(
        "cargo binstall --no-confirm --locked \"cargo-nextest@${CARGO_NEXTEST_VERSION}\""
    ));
    assert!(PROPERTY_TESTS_WORKFLOW.contains("~/.cargo/bin/cargo-nextest"));
    assert!(PROPERTY_TESTS_WORKFLOW.contains("~/.cache/cargo-binstall"));
    assert!(PROPERTY_TESTS_WORKFLOW.contains(
        "cargo-nextest-${{ runner.os }}-${{ runner.arch }}-${{ env.CARGO_NEXTEST_VERSION }}"
    ));
}
