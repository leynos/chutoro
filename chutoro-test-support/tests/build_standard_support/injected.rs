//! The policy checks fed canned `make -n` text through the injected runner, so no
//! process runs. The real-`make` tests live in the contract file; these cover the
//! parsing and policy path, and run a synthetic held-out target so the check fires
//! in a repository that defines none.

use std::fmt::Write as _;

use super::{
    config::{Pin, THREADS_FLAG},
    make::{
        Host, MakeRunner, Target, development_problems, held_out_problems, held_out_problems_for,
        held_out_target_count,
    },
};

/// Renders canned `make -n` text for a fake runner through a fallible writer, so
/// the fakes keep the runner's `Result` shape honestly.
fn canned(text: std::fmt::Arguments) -> Result<String, String> {
    let mut out = String::new();
    out.write_fmt(text).map_err(|error| error.to_string())?;
    Ok(out)
}

/// Returns the linker flag the standard adds on a host: mold on Linux, nothing elsewhere.
const fn linker_flag(host: Host) -> &'static str {
    if host.takes_linker_flag() {
        " -Clink-arg=-fuse-ld=mold"
    } else {
        ""
    }
}

/// A fake runner: a compliant `make -n` for any target, with no process behind it.
fn compliant_make(_target: Target<'_>, host: Host) -> Result<String, String> {
    let linker = linker_flag(host);
    canned(format_args!(
        "RUSTFLAGS=\"${{RUSTFLAGS:+$RUSTFLAGS }}-D warnings {THREADS_FLAG}{linker}\" cargo test\n"
    ))
}

/// A fake runner whose command loses the caller's `RUSTFLAGS`.
fn dropping_make(_target: Target<'_>, _host: Host) -> Result<String, String> {
    canned(format_args!("RUSTFLAGS=\"-D warnings\" cargo test\n"))
}

/// A fake runner whose test command assigns nothing, so the warning policy never reaches it.
fn bare_test_make(_target: Target<'_>, _host: Host) -> Result<String, String> {
    canned(format_args!("cargo test\n"))
}

/// A fake runner whose lint commands assign nothing beside a command that does.
fn bare_lint_make(_target: Target<'_>, host: Host) -> Result<String, String> {
    let linker = linker_flag(host);
    canned(format_args!(
        "RUSTFLAGS=\"${{RUSTFLAGS:+$RUSTFLAGS }}-D warnings {THREADS_FLAG}{linker}\" cargo test\ncargo clippy --all-targets\nwhitaker --all\n"
    ))
}

/// A fake runner with commands that run no compiled code under test, beside one that assigns.
fn exempt_commands_make(_target: Target<'_>, host: Host) -> Result<String, String> {
    let linker = linker_flag(host);
    canned(format_args!(
        "RUSTFLAGS=\"${{RUSTFLAGS:+$RUSTFLAGS }}-D warnings {THREADS_FLAG}{linker}\" cargo test\ncargo fmt --all --check\ncargo metadata --format-version 1\nRUSTDOCFLAGS=\"-D warnings\" cargo doc\n"
    ))
}

/// A fake runner whose command assigns an empty `RUSTFLAGS`, dropping the caller's.
fn release_clearing_make(_target: Target<'_>, _host: Host) -> Result<String, String> {
    canned(format_args!("RUSTFLAGS=\"\" cargo build --release\n"))
}

/// A fake runner whose held-out target runs an inspection command beside an assigning build.
fn held_out_inspecting_make(_target: Target<'_>, _host: Host) -> Result<String, String> {
    canned(format_args!(
        "cargo metadata --format-version 1 --locked\nRUSTFLAGS=\"-D warnings\" cargo build --release\n"
    ))
}

/// A fake runner whose chained recipe leaves its second command bare beside a probe that assigns.
fn chained_bare_make(_target: Target<'_>, host: Host) -> Result<String, String> {
    let linker = linker_flag(host);
    canned(format_args!(
        "if RUSTFLAGS=\"${{RUSTFLAGS:+$RUSTFLAGS }}-D warnings {THREADS_FLAG}{linker}\" cargo nextest --version; then cargo nextest run; else echo \"falling back to cargo test\"; RUSTFLAGS=\"${{RUSTFLAGS:+$RUSTFLAGS }}-D warnings {THREADS_FLAG}{linker}\" cargo test; fi\n"
    ))
}

/// A fake runner whose held-out command assigns `RUSTFLAGS` without a standard flag.
fn held_out_assigning_make(_target: Target<'_>, _host: Host) -> Result<String, String> {
    canned(format_args!(
        "RUSTFLAGS=\"-D warnings\" cargo build --release\n"
    ))
}

/// A fake runner whose held-out command assigns nothing, so it takes the configuration's flags.
fn held_out_unassigned_make(_target: Target<'_>, _host: Host) -> Result<String, String> {
    canned(format_args!("cargo build --release\n"))
}

/// A fake runner for a target that is not defined.
fn undefined_make(target: Target<'_>, _host: Host) -> Result<String, String> {
    Err(format!(
        "`make -n {}` failed, so it is not defined",
        target.name()
    ))
}

/// Turns a failed expectation into the error a test returns.
fn ensure(holds: bool, message: &str) -> Result<(), String> {
    if holds {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

/// Scenario: the development policy fed canned `make -n` text through the injected runner, so
/// no process runs.
///
/// Invariant: a compliant command raises no complaint on either host, a command that drops the
/// caller's flags raises one, and a runner error reaches the caller instead of being read as an
/// empty output.
#[test]
fn the_development_policy_runs_against_an_injected_runner() -> Result<(), String> {
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
    )
}

/// Scenario: development recipes with commands that assign no `RUSTFLAGS`, chained and alone,
/// beside commands that need none.
///
/// Invariant: a build, test or lint command that assigns nothing is refused wherever it sits,
/// including beside a probe that does assign; a formatter, a metadata probe and a documentation
/// build are not refused.
#[test]
fn a_development_command_without_an_assignment_is_refused_and_an_exempt_one_is_not()
-> Result<(), String> {
    let pin = Pin::Nightly;
    let (bare_test, _) = development_problems(bare_test_make, Host::Linux, pin)?;
    ensure(
        !bare_test.is_empty(),
        "a test command that assigns no RUSTFLAGS passed",
    )?;
    let (bare_lint, _) = development_problems(bare_lint_make, Host::Linux, pin)?;
    ensure(
        !bare_lint.is_empty(),
        "lint commands that assign no RUSTFLAGS passed",
    )?;
    let (chained, _) = development_problems(chained_bare_make, Host::Linux, pin)?;
    ensure(
        !chained.is_empty()
            && chained
                .iter()
                .all(|problem| problem.contains("nextest run")),
        &format!("a bare command chained beside an assigning probe raised {chained:?}"),
    )?;
    let (exempt, exempt_read) = development_problems(exempt_commands_make, Host::Linux, pin)?;
    ensure(
        exempt.is_empty() && exempt_read > 0,
        &format!("a formatter, probe or doc build raised {exempt:?}"),
    )
}

/// Defines a fake runner that prints one release command, whatever the target and host.
macro_rules! release_make {
    ($name:ident, $text:expr) => {
        fn $name(_target: Target<'_>, _host: Host) -> Result<String, String> {
            canned(format_args!("{}\n", $text))
        }
    };
}

release_make!(
    keeps_dash,
    "RUSTFLAGS=\"${RUSTFLAGS-}\" cargo build --release"
);
release_make!(
    keeps_dash_with_flags,
    "RUSTFLAGS=\"${RUSTFLAGS-} -D warnings\" cargo build --release"
);
release_make!(
    keeps_plus,
    "RUSTFLAGS=\"${RUSTFLAGS:+$RUSTFLAGS }\" cargo build --release"
);
release_make!(
    keeps_plus_with_flags,
    "RUSTFLAGS=\"${RUSTFLAGS:+$RUSTFLAGS }-Zpolonius=next\" cargo build --release"
);
release_make!(assigns_nothing_unquoted, "RUSTFLAGS= cargo build --release");
release_make!(
    assigns_an_empty_string,
    "RUSTFLAGS=\"\" cargo build --release"
);
release_make!(
    assigns_the_standard_flags_alone,
    "RUSTFLAGS=-Zthreads=8 -Clink-arg=-fuse-ld=mold cargo build --release"
);
release_make!(
    assigns_its_own_flags_alone,
    "RUSTFLAGS=\"-D warnings\" cargo build --release"
);

/// A release command form, and whether the contract must accept it.
struct ReleaseForm {
    text: &'static str,
    runner: MakeRunner,
    accepted: bool,
}

/// Every release form the contract must tell apart: those that keep the caller's `RUSTFLAGS`, and
/// those that clear, replace or garble them.
const RELEASE_FORMS: [ReleaseForm; 8] = [
    ReleaseForm {
        text: "${RUSTFLAGS-}",
        runner: keeps_dash,
        accepted: true,
    },
    ReleaseForm {
        text: "${RUSTFLAGS-} plus flags",
        runner: keeps_dash_with_flags,
        accepted: true,
    },
    ReleaseForm {
        text: "${RUSTFLAGS:+$RUSTFLAGS }",
        runner: keeps_plus,
        accepted: true,
    },
    ReleaseForm {
        text: "${RUSTFLAGS:+$RUSTFLAGS } plus flags",
        runner: keeps_plus_with_flags,
        accepted: true,
    },
    ReleaseForm {
        text: "RUSTFLAGS=",
        runner: assigns_nothing_unquoted,
        accepted: false,
    },
    ReleaseForm {
        text: "RUSTFLAGS=\"\"",
        runner: assigns_an_empty_string,
        accepted: false,
    },
    ReleaseForm {
        text: "RUSTFLAGS=<standard flags> alone",
        runner: assigns_the_standard_flags_alone,
        accepted: false,
    },
    ReleaseForm {
        text: "RUSTFLAGS=\"-D warnings\" alone",
        runner: assigns_its_own_flags_alone,
        accepted: false,
    },
];

/// Scenario: a release command in each form it can take, fed through the injected runner.
///
/// Invariant: a release command is accepted exactly when it assigns `RUSTFLAGS` in a form that keeps
/// the caller's value, with or without flags of its own; an empty, unquoted, standard-only or
/// own-flags-only assignment is refused, whether as a complaint or as an unreadable line.
#[test]
fn a_release_command_is_accepted_only_when_it_keeps_the_callers_rustflags() -> Result<(), String> {
    for form in RELEASE_FORMS {
        let accepted = matches!(held_out_problems_for(form.runner, &["release"]), Ok((problems, _)) if problems.is_empty());
        ensure(
            accepted == form.accepted,
            &format!(
                "`{}` was judged accepted={accepted}, not {}",
                form.text, form.accepted
            ),
        )?;
    }
    Ok(())
}

/// Scenario: held-out targets fed canned `make -n` text, through a synthetic target so the check
/// fires in a repository that defines none.
///
/// Invariant: a held-out command must assign `RUSTFLAGS`; a release build must keep the caller's
/// flags while a coverage build, a measurement, need not; an inspection command is not refused; a
/// runner error reaches the caller.
#[test]
fn the_held_out_policy_runs_against_an_injected_runner() -> Result<(), String> {
    ensure(
        held_out_problems_for(undefined_make, &["synthetic"]).is_err(),
        "a held-out runner error was swallowed",
    )?;
    let (clean, read) = held_out_problems_for(held_out_assigning_make, &["synthetic"])?;
    ensure(
        clean.is_empty() && read == 1,
        &format!("an assigning held-out command raised {clean:?}"),
    )?;
    let (measuring, _) = held_out_problems_for(release_clearing_make, &["coverage"])?;
    ensure(
        measuring.is_empty(),
        &format!("a coverage build that ignores the caller's RUSTFLAGS raised {measuring:?}"),
    )?;
    let (inspecting, _) = held_out_problems_for(held_out_inspecting_make, &["synthetic"])?;
    ensure(
        inspecting.is_empty(),
        &format!("an inspection command in a held-out target raised {inspecting:?}"),
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
