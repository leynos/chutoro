//! Detect whether compile-contract tests are built with nightly Rust.

use std::{
    env,
    error::Error,
    ffi::OsString,
    io::{self, Write},
    process::Command,
};

fn main() -> Result<(), Box<dyn Error>> {
    emit_cargo_directive("cargo:rerun-if-changed=build.rs")?;
    emit_cargo_directive("cargo:rerun-if-env-changed=RUSTC")?;
    emit_cargo_directive("cargo:rustc-check-cfg=cfg(nightly)")?;

    if is_nightly_compiler() {
        emit_cargo_directive("cargo:rustc-cfg=nightly")?;
    }
    Ok(())
}

/// Writes one build-script directive for Cargo to consume.
fn emit_cargo_directive(directive: &str) -> io::Result<()> {
    writeln!(io::stdout().lock(), "{directive}")
}

/// Reports whether the configured compiler identifies itself as nightly.
fn is_nightly_compiler() -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .is_some_and(|version| version.contains("nightly"))
}
