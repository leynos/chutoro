//! Compile-time checks for the public `ClusteringResult` API surface.

#[cfg(feature = "cpu")]
use std::{path::Path, process::Command, str};

/// Confirms the CPU-only panicking constructor remains outside the public API.
#[cfg(feature = "cpu")]
#[test]
fn clustering_result_panicking_constructor_is_private_when_cpu_enabled() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/clustering_result_from_assignments_private.rs");
}

/// Confirms disabling CPU removes the CPU-only result constructor entirely.
#[cfg(feature = "cpu")]
#[test]
fn clustering_result_api_is_checked_without_cpu() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("clustering_result_without_cpu");
    let target_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate manifest directory must be inside the workspace")
        .join("target")
        .join("result-api-surface-no-cpu");

    let output = Command::new(env!("CARGO"))
        .arg("check")
        .arg("--manifest-path")
        .arg(fixture_dir.join("Cargo.toml"))
        .arg("--quiet")
        .env("CARGO_TARGET_DIR", target_dir)
        .output()
        .expect("cpu-disabled result API check should run");

    assert!(
        !output.status.success(),
        "cpu-disabled result API fixture unexpectedly compiled",
    );

    let stderr = str::from_utf8(&output.stderr).expect("cargo stderr must be utf-8");
    assert!(
        stderr.contains("no function or associated item named `from_assignments`"),
        "cpu-disabled result API fixture did not reject `from_assignments`:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
}
