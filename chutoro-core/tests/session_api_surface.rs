//! Compile-time checks for the CPU-gated session API surface.

use std::{env, path::Path, process::Command, str};

#[test]
#[cfg(feature = "cpu")]
fn session_api_compiles_when_cpu_feature_is_enabled() {
    let cases = trybuild::TestCases::new();
    cases.pass("tests/trybuild/session_api_cpu_enabled.rs");
    cases.compile_fail("tests/trybuild/session_api_non_send_sync_source.rs");
}

#[test]
fn session_api_is_unavailable_without_cpu_feature() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("session_api_without_cpu");
    let target_dir = env::temp_dir().join(format!("chutoro-core-no-cpu-{}", std::process::id()));

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
