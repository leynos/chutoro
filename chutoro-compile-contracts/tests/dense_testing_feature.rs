//! Compile-time checks for the dense provider's opt-in testing feature.

use camino::Utf8Path;
use std::process::{Command, Output};

/// Confirms downstream crates can import dense fixtures when the feature is
/// enabled.
#[test]
fn dense_testing_module_is_available_with_the_feature() {
    let output = check_fixture("dense_testing_with_feature")
        .expect("feature-enabled fixture check must start");
    assert!(
        output.status.success(),
        "feature-enabled fixture must compile: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Confirms the dense fixture module stays unavailable without its opt-in
/// feature.
#[test]
fn dense_testing_module_is_unavailable_without_the_feature() {
    let output = check_fixture("dense_testing_without_feature")
        .expect("feature-disabled fixture check must start");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && stderr.contains("chutoro_providers_dense::testing"),
        "featureless fixture must fail at the gated import: {stderr}"
    );
}

/// Checks one downstream fixture crate with an isolated Cargo target directory.
fn check_fixture(fixture: &str) -> std::io::Result<Output> {
    let package_dir = Utf8Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = package_dir.join(format!("tests/fixtures/{fixture}/Cargo.toml"));
    let target_dir = package_dir.join("../target/dense-testing-feature-contracts");

    Command::new(env!("CARGO"))
        .args(["check", "--offline", "--locked", "--manifest-path"])
        .arg(manifest.as_str())
        .env("CARGO_TARGET_DIR", target_dir.as_str())
        .output()
}
