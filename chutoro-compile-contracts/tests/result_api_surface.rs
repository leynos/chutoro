//! Compile-time checks for the public `ClusteringResult` API surface.

#[cfg(feature = "cpu")]
use std::{path::Path, process::Command};

#[cfg(feature = "cpu")]
#[test]
fn clustering_result_panicking_constructor_is_private_when_cpu_enabled() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/clustering_result_from_assignments_private.rs");
}

#[cfg(not(feature = "cpu"))]
#[test]
fn clustering_result_panicking_constructor_is_unavailable_without_cpu() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/trybuild/clustering_result_from_assignments_unavailable.rs");
}

#[cfg(feature = "cpu")]
#[test]
fn clustering_result_api_is_checked_without_cpu() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate manifest directory must be inside the workspace");
    let target_dir = workspace_root
        .join("target")
        .join("result-api-surface-no-cpu");

    let output = Command::new(env!("CARGO"))
        .current_dir(workspace_root)
        .args([
            "test",
            "-p",
            "chutoro-compile-contracts",
            "--no-default-features",
            "--test",
            "result_api_surface",
            "--",
            "--exact",
            "clustering_result_panicking_constructor_is_unavailable_without_cpu",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .env("RUSTFLAGS", "-D warnings")
        .output()
        .expect("cpu-disabled result API check should run");

    assert!(
        output.status.success(),
        "cpu-disabled result API check failed:\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
}
