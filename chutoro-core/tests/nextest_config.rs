//! Regression checks for the repository's nextest profile configuration.

use rstest::rstest;

const NEXTEST_CONFIG: &str = include_str!("../../.config/nextest.toml");
const PROPERTY_TESTS_WORKFLOW: &str = include_str!("../../.github/workflows/property-tests.yml");
const BENCH_SLOW_TIMEOUT: &str =
    "slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }";

fn default_override_blocks() -> Vec<&'static str> {
    NEXTEST_CONFIG
        .split("[[profile.default.overrides]]")
        .skip(1)
        .map(str::trim)
        .collect()
}

fn workflow_job_block(job: &str) -> Result<&'static str, String> {
    let (_, rest) = PROPERTY_TESTS_WORKFLOW
        .split_once(&format!("  {job}:"))
        .ok_or_else(|| format!("workflow job '{job}' not found"))?;
    let block = match rest.split_once("\n\n  ") {
        Some((block, _)) => block,
        None => rest,
    };

    Ok(block)
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
    let override_blocks = default_override_blocks();
    let idempotency_override_present = override_blocks.into_iter().any(|block| {
        block.contains("filter = \"test(/hnsw_idempotency_preserved_proptest/)\"")
            && block.contains(BENCH_SLOW_TIMEOUT)
    });
    assert!(idempotency_override_present);

    let pr_job = workflow_job_block("property-tests-pr").expect("property-tests-pr job must exist");
    assert!(pr_job.contains("timeout-minutes: 20"));
}
