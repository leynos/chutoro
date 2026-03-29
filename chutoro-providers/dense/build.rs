//! Detect whether Cargo is compiling this crate with a nightly Rust toolchain.

use std::env;
use std::ffi::OsString;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rustc-check-cfg=cfg(nightly)");

    if is_nightly_compiler() {
        println!("cargo:rustc-cfg=nightly");
    }
}

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
