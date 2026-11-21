//! Compile-time checks for the `http`-gated staging lock.

use camino::Utf8Path;
use std::process::{Command, Output};

/// Confirms downstream crates can import the staging lock when the feature is
/// enabled.
#[test]
fn staging_lock_is_available_with_the_http_feature() {
    let output =
        check_fixture("staging_lock_with_http").expect("feature-enabled fixture check must start");
    assert!(
        output.status.success(),
        "feature-enabled fixture must compile: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Confirms the staging lock stays unavailable without its opt-in feature.
#[test]
fn staging_lock_is_unavailable_without_the_http_feature() {
    let output = check_fixture("staging_lock_without_http")
        .expect("feature-disabled fixture check must start");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && stderr.contains("StagingLock"),
        "featureless fixture must fail at the gated import: {stderr}"
    );
}

/// Check one downstream fixture crate with an isolated Cargo target directory.
fn check_fixture(fixture: &str) -> std::io::Result<Output> {
    let package_dir = Utf8Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = package_dir.join(format!("tests/fixtures/{fixture}/Cargo.toml"));
    let target_dir = package_dir.join("../target/staging-lock-feature-contracts");

    Command::new(env!("CARGO"))
        .args(["check", "--offline", "--locked", "--manifest-path"])
        .arg(manifest.as_str())
        .env("CARGO_TARGET_DIR", target_dir.as_str())
        .output()
}
