//! Regression checks for the repository's nextest profile configuration.

const NEXTEST_CONFIG: &str = include_str!("../../.config/nextest.toml");

#[test]
fn nextest_default_profile_keeps_benchmark_timeout_guards() {
    assert!(NEXTEST_CONFIG.contains("global-timeout = \"40m\""));
    assert!(NEXTEST_CONFIG.contains("filter = \"package(chutoro-benches) & kind(bench)\""));
    assert!(NEXTEST_CONFIG.contains("threads-required = 8"));
    assert!(NEXTEST_CONFIG.contains(
        "slow-timeout = { period = \"600s\", terminate-after = 1, grace-period = \"5s\" }"
    ));
    assert!(
        NEXTEST_CONFIG
            .contains("filter = \"package(chutoro-benches) & test(/extract_labels\\\\//)\"")
    );
    assert!(NEXTEST_CONFIG.contains(
        "filter = \"package(chutoro-benches) & test(/edge_harvest_construction\\\\//)\""
    ));
}
