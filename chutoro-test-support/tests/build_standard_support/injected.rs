//! The policy checks fed canned `make -n` text through the injected runner, so no
//! process runs. The real-`make` tests live in the contract file; these cover the
//! parsing and policy path, and run a synthetic held-out target so the check fires
//! in a repository that defines none.

use std::fmt::Write as _;

use super::{
    config::{Pin, THREADS_FLAG},
    make::{
        Host, development_problems, held_out_problems, held_out_problems_for, held_out_target_count,
    },
};

/// Renders canned `make -n` text for a fake runner through a fallible writer, so
/// the fakes keep the runner's `Result` shape honestly.
fn canned(text: std::fmt::Arguments) -> Result<String, String> {
    let mut out = String::new();
    out.write_fmt(text).map_err(|error| error.to_string())?;
    Ok(out)
}

/// A fake runner: a compliant `make -n` for any target, with no process behind it.
fn compliant_make(_target: &str, host: Host) -> Result<String, String> {
    let linker = if host.takes_linker_flag() {
        " -Clink-arg=-fuse-ld=mold"
    } else {
        ""
    };
    canned(format_args!(
        "RUSTFLAGS=\"${{RUSTFLAGS:+$RUSTFLAGS }}-D warnings {THREADS_FLAG}{linker}\" cargo test\n"
    ))
}

/// A fake runner whose command loses the caller's `RUSTFLAGS`.
fn dropping_make(_target: &str, _host: Host) -> Result<String, String> {
    canned(format_args!("RUSTFLAGS=\"-D warnings\" cargo test\n"))
}

/// A fake runner whose held-out command assigns `RUSTFLAGS` without a standard flag.
fn held_out_assigning_make(_target: &str, _host: Host) -> Result<String, String> {
    canned(format_args!(
        "RUSTFLAGS=\"-D warnings\" cargo build --release\n"
    ))
}

/// A fake runner whose held-out command assigns nothing, so it takes the configuration's flags.
fn held_out_unassigned_make(_target: &str, _host: Host) -> Result<String, String> {
    canned(format_args!("cargo build --release\n"))
}

/// A fake runner for a target that is not defined.
fn undefined_make(target: &str, _host: Host) -> Result<String, String> {
    Err(format!("`make -n {target}` failed, so it is not defined"))
}

/// Turns a failed expectation into the error a test returns.
fn ensure(holds: bool, message: &str) -> Result<(), String> {
    if holds {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

/// Scenario: the policy checks fed canned `make -n` text through the injected
/// runner, so no process runs.
///
/// Invariant: a compliant command raises no complaint on either host, a command
/// that drops the caller's flags raises one per target, and a runner error
/// reaches the caller instead of being read as an empty output.
#[test]
fn the_policy_checks_run_against_an_injected_runner() -> Result<(), String> {
    let pin = Pin::Nightly;
    for host in [Host::Linux, Host::Darwin] {
        let (problems, read) = development_problems(compliant_make, host, pin)?;
        ensure(
            problems.is_empty(),
            &format!("a compliant fake raised {problems:?}"),
        )?;
        ensure(read > 0, "the fake's commands were not read")?;
    }
    let (dropped, _) = development_problems(dropping_make, Host::Linux, pin)?;
    ensure(
        !dropped.is_empty(),
        "a command that drops the caller's flags passed",
    )?;
    ensure(
        development_problems(undefined_make, Host::Linux, pin).is_err(),
        "a runner error was swallowed",
    )?;
    // A synthetic held-out target runs the check in every repository, including one that defines none.
    ensure(
        held_out_problems_for(undefined_make, &["synthetic"]).is_err(),
        "a held-out runner error was swallowed",
    )?;
    let (clean, read) = held_out_problems_for(held_out_assigning_make, &["synthetic"])?;
    ensure(
        clean.is_empty() && read == 1,
        &format!("an assigning held-out command raised {clean:?}"),
    )?;
    let (unassigned, _) = held_out_problems_for(held_out_unassigned_make, &["synthetic"])?;
    ensure(
        !unassigned.is_empty(),
        "a held-out command that assigns no RUSTFLAGS passed",
    )?;
    ensure(
        held_out_target_count() == 0 || held_out_problems(undefined_make).is_err(),
        "the listed held-out targets were not run",
    )
}
