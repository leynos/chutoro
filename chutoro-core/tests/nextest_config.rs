//! Regression checks for the repository's nextest profile configuration.

use rstest::rstest;

const NEXTEST_CONFIG: &str = include_str!("../../.config/nextest.toml");
const PROPERTY_TESTS_WORKFLOW: &str = include_str!("../../.github/workflows/property-tests.yml");
const BENCHMARK_REGRESSIONS_WORKFLOW: &str =
    include_str!("../../.github/workflows/benchmark-regressions.yml");
const MAKEFILE: &str = include_str!("../../Makefile");
const BENCH_SLOW_TIMEOUT: &str =
    "slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }";
const TRYBUILD_SLOW_TIMEOUT: &str =
    "slow-timeout = { period = \"300s\", terminate-after = 1, grace-period = \"5s\" }";

fn default_override_blocks() -> Vec<&'static str> {
    override_blocks("default")
}

fn ci_override_blocks() -> Vec<&'static str> {
    override_blocks("ci")
}

fn override_blocks(profile_name: &str) -> Vec<&'static str> {
    NEXTEST_CONFIG
        .split(&format!("[[profile.{profile_name}.overrides]]"))
        .skip(1)
        .map(str::trim)
        .collect()
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
#[case("filter = \"package(chutoro-benches) & kind(bench)\"")]
#[case("filter = \"package(chutoro-benches) & test(/extract_labels\\\\//)\"")]
#[case("filter = \"package(chutoro-benches) & test(/edge_harvest_construction\\\\//)\"")]
fn nextest_default_profile_keeps_benchmark_timeout_guards(#[case] filter_value: &str) {
    let override_blocks = default_override_blocks();
    let override_present = override_blocks.into_iter().any(|block| {
        block.contains(filter_value)
            && block.contains("threads-required = 8")
            && block.contains(BENCH_SLOW_TIMEOUT)
    });
    assert!(override_present);
}

#[test]
fn property_tests_pr_timeout_covers_hnsw_idempotency_budget() {
    let override_blocks = ci_override_blocks();
    let idempotency_override_present = override_blocks.into_iter().any(|block| {
        block.contains("filter = \"test(/hnsw_idempotency_preserved_proptest/)\"")
            && block.contains(BENCH_SLOW_TIMEOUT)
    });
    assert!(idempotency_override_present);

    let pr_job = workflow_job_block("property-tests-pr").expect("property-tests-pr job must exist");
    assert!(pr_job.contains("timeout-minutes: 20"));
}

#[rstest]
#[case("default")]
#[case("ci")]
fn nextest_profiles_keep_trybuild_timeout_guards(#[case] profile_name: &str) {
    let override_blocks = override_blocks(profile_name);
    let override_present = override_blocks.into_iter().any(|block| {
        block.contains("portable_simd_gating_compile_checks")
            && block.contains("session_api_compiles_when_cpu_feature_is_enabled")
            && block.contains("threads-required = 4")
            && block.contains(TRYBUILD_SLOW_TIMEOUT)
    });
    assert!(override_present);
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
    assert!(smoke_job.contains(
        "cargo bench -p chutoro-benches --bench hnsw -- hnsw_build/n=100,M=8,ef=16 --exact"
    ));
}

#[rstest]
#[case("property-tests-pr")]
#[case("property-tests-weekly")]
fn property_tests_use_eight_core_runner(#[case] job: &str) {
    let job_block = workflow_job_block(job).expect("property test job must exist");
    assert!(job_block.contains("runs-on: ubicloud-standard-8"));
}
