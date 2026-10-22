//! Compile-time tests for nightly portable-SIMD gating contracts.
//!
//! These compile-contract tests verify the stable/nightly feature matrix at
//! compile time for the portable-SIMD backend.

/// Confirms stable builds reject portable SIMD without its opt-in feature.
#[test]
#[cfg(not(nightly))]
fn portable_simd_is_rejected_without_feature() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = manifest_dir.join("tests/fixtures/portable_simd_without_feature/Cargo.toml");
    let target_dir = manifest_dir
        .parent()
        .expect("compile-contract package must be inside the workspace")
        .join("target/portable-simd-stable-without-feature");
    let output = std::process::Command::new(env!("CARGO"))
        .args(["check", "--offline", "--locked", "--manifest-path"])
        .arg(manifest)
        .env("CARGO_TARGET_DIR", target_dir)
        .env_remove("RUSTC_BOOTSTRAP")
        .output()
        .expect("stable no-feature compile-contract check should run");
    let stderr = std::str::from_utf8(&output.stderr).expect("Cargo stderr must be UTF-8");
    assert!(
        !output.status.success() && stderr.contains("E0658") && stderr.contains("portable_simd"),
        "stable must reject portable SIMD without its feature gate: {stderr}"
    );
}

/// Confirms nightly still rejects the existing fixture without its opt-in gate.
///
/// Cargo diagnostics are checked by error code because nightly adds feature-help
/// lines that differ from the stable Trybuild snapshot.
#[test]
#[cfg(all(nightly, not(feature = "nightly_portable_simd")))]
fn portable_simd_is_rejected_on_nightly_without_feature() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = manifest_dir.join("tests/fixtures/portable_simd_without_feature/Cargo.toml");
    let target_dir = manifest_dir
        .parent()
        .expect("compile-contract package must be inside the workspace")
        .join("target/portable-simd-without-feature");
    let output = std::process::Command::new(env!("CARGO"))
        .args(["check", "--offline", "--locked", "--manifest-path"])
        .arg(manifest)
        .env("CARGO_TARGET_DIR", target_dir)
        .output()
        .expect("nightly no-feature compile-contract check should run");
    let stderr = std::str::from_utf8(&output.stderr).expect("Cargo stderr must be UTF-8");
    assert!(
        !output.status.success() && stderr.contains("E0658") && stderr.contains("portable_simd"),
        "nightly must reject portable SIMD without its feature gate: {stderr}"
    );
}

/// Confirms nightly builds accept portable SIMD when its feature is enabled.
#[test]
#[cfg(all(nightly, feature = "nightly_portable_simd"))]
fn portable_simd_compiles_with_feature() {
    let t = trybuild::TestCases::new();
    t.pass("tests/trybuild/portable_simd_with_feature.rs");
}
