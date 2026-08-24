//! Detect whether Cargo is compiling this crate with a nightly Rust toolchain.

use mockable::{DefaultEnv, Env};
use std::error::Error;
use std::ffi::OsString;
use std::io::{self, Write};
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    emit_cargo_directive("cargo:rerun-if-changed=build.rs")?;
    emit_cargo_directive("cargo:rerun-if-env-changed=RUSTC")?;
    emit_cargo_directive("cargo:rustc-check-cfg=cfg(kani)")?;
    emit_cargo_directive("cargo:rustc-check-cfg=cfg(nightly)")?;

    if is_nightly_compiler(&DefaultEnv) {
        emit_cargo_directive("cargo:rustc-cfg=nightly")?;
    }
    Ok(())
}

/// Write one build-script directive for Cargo to consume.
fn emit_cargo_directive(directive: &str) -> io::Result<()> {
    writeln!(io::stdout().lock(), "{directive}")
}

/// Report whether the configured Rust compiler identifies itself as nightly.
fn is_nightly_compiler(env: &dyn Env) -> bool {
    let rustc = env
        .os_string("RUSTC")
        .unwrap_or_else(|| OsString::from("rustc"));
    Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .is_some_and(|version| version.contains("nightly"))
}
