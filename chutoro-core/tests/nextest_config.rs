//! Regression checks for the repository's nextest profile configuration.

const NEXTEST_CONFIG: &str = include_str!("../../.config/nextest.toml");
const PROPERTY_TESTS_WORKFLOW: &str = include_str!("../../.github/workflows/property-tests.yml");

fn default_override_blocks() -> Vec<&'static str> {
    NEXTEST_CONFIG
        .split("[[profile.default.overrides]]")
        .skip(1)
        .map(str::trim)
        .collect()
}

fn workflow_job_block(job: &str) -> &'static str {
    PROPERTY_TESTS_WORKFLOW
        .split_once(&format!("  {job}:"))
        .and_then(|(_, rest)| rest.split_once("\n\n  "))
        .map_or(PROPERTY_TESTS_WORKFLOW, |(block, _)| block)
}

#[test]
fn nextest_default_profile_keeps_benchmark_timeout_guards() {
    assert!(NEXTEST_CONFIG.contains("global-timeout = \"40m\""));

    let bench_override_present = default_override_blocks().into_iter().any(|block| {
        block.contains("filter = \"package(chutoro-benches) & kind(bench)\"")
            && block.contains("threads-required = 8")
            && block.contains(
                "slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }",
            )
    });
    assert!(bench_override_present);

    let extraction_override_present = default_override_blocks().into_iter().any(|block| {
        block.contains("filter = \"package(chutoro-benches) & test(/extract_labels\\\\//)\"")
            && block.contains("threads-required = 8")
            && block.contains(
                "slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }",
            )
    });
    assert!(extraction_override_present);

    let edge_harvest_override_present = default_override_blocks().into_iter().any(|block| {
        block.contains(
            "filter = \"package(chutoro-benches) & test(/edge_harvest_construction\\\\//)\"",
        ) && block.contains("threads-required = 8")
            && block.contains(
                "slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }",
            )
    });
    assert!(edge_harvest_override_present);
}

#[test]
fn property_tests_pr_timeout_covers_hnsw_idempotency_budget() {
    assert!(NEXTEST_CONFIG.contains(
        "filter = \"test(/hnsw_idempotency_preserved_proptest/)\"\n\
         slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }",
    ));

    let pr_job = workflow_job_block("property-tests-pr");
    assert!(pr_job.contains("timeout-minutes: 20"));
}
