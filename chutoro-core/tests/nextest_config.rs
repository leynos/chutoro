//! Regression checks for the repository's nextest profile configuration.

const NEXTEST_CONFIG: &str = include_str!("../../.config/nextest.toml");

fn default_override_blocks() -> Vec<&'static str> {
    NEXTEST_CONFIG
        .split("[[profile.default.overrides]]")
        .skip(1)
        .map(str::trim)
        .collect()
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
