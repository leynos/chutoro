//! Compile-time contract for the CPU-gated MST scenario target.
//!
//! `chutoro-core/Cargo.toml` declares `required-features = ["cpu"]` for the
//! `mst_edge_preparation_bdd` target, because the scenario suite imports
//! `parallel_kruskal` and the other exports the crate only offers with `cpu`.
//! Without the declaration a `--no-default-features` build tried to compile
//! the target and failed to resolve its imports.
//!
//! The declaration is observable through Cargo's own target selection: naming
//! a target whose required features are disabled has Cargo refuse it by name,
//! rather than compiling it. That message is the contract asserted here. The
//! import-level gating behind it is the same mechanism the sibling
//! `session_api_surface` contract pins for the session API.

use std::process::Command;

/// Verifies Cargo refuses the scenario target when `cpu` is disabled.
#[test]
fn mst_scenario_target_requires_the_cpu_feature() {
    let output = Command::new(env!("CARGO"))
        .args([
            "test",
            "-p",
            "chutoro-core",
            "--no-default-features",
            "--test",
            "mst_edge_preparation_bdd",
            "--no-run",
            "--offline",
            "--locked",
        ])
        .env("CARGO_NET_OFFLINE", "true")
        .output()
        .expect("cargo should run");

    assert!(
        !output.status.success(),
        "Cargo compiled a target whose required features are disabled"
    );

    let stderr = String::from_utf8(output.stderr).expect("cargo stderr must be utf-8");
    assert!(
        stderr.contains(
            "target `mst_edge_preparation_bdd` in package `chutoro-core` requires the features: \
             `cpu`"
        ),
        "Cargo did not name the gated target and feature; stderr was:\n{stderr}"
    );
}
