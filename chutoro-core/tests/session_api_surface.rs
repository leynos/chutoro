//! Compile-time checks for the CPU-gated session API surface.

use std::{
    path::{Path, PathBuf},
    process::Command,
    str,
};

/// Returns the workspace root containing this crate's manifest directory.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|directory| directory.join("Cargo.toml").is_file() && directory.join(".git").exists())
        .map_or_else(
            || panic!("crate manifest directory must be inside the workspace"),
            Path::to_path_buf,
        )
}

#[test]
#[cfg(feature = "cpu")]
fn session_api_compiles_when_cpu_feature_is_enabled() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/public_const_apis.rs");
    cases.pass("tests/trybuild/session_api_cpu_enabled.rs");
    cases.compile_fail("tests/trybuild/session_api_non_send_sync_source.rs");
}

#[test]
fn session_api_is_unavailable_without_cpu_feature() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("session_api_without_cpu");
    let target_dir = workspace_root().join("target").join("session-api-surface");

    let output = Command::new(env!("CARGO"))
        .arg("check")
        .arg("--manifest-path")
        .arg(fixture_dir.join("Cargo.toml"))
        .arg("--no-default-features")
        .arg("--quiet")
        .env("CARGO_TARGET_DIR", &target_dir)
        .output()
        .expect("cargo check should run");

    assert!(
        !output.status.success(),
        "cpu-disabled fixture unexpectedly compiled"
    );

    let stderr = str::from_utf8(&output.stderr).expect("cargo stderr must be utf-8");
    assert!(stderr.contains("cannot find type `SessionConfig` in crate `chutoro_core`"));
    assert!(stderr.contains("could not find `SessionRefreshPolicy` in `chutoro_core`"));
    assert!(stderr.contains("cannot find type `ClusteringSession` in crate `chutoro_core`"));
    assert!(stderr.contains("no method named `build_session`"));
}
