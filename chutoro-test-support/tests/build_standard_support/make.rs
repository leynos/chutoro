//! Readers for the Makefile half of the build standard: the commands `make -n`
//! prints for each development, coverage and release target, judged against a
//! toolchain pin and a host.

use std::process::Command;

use super::config::{Flags, LINKER_FLAG, Pin, Problems, THREADS_FLAG};

/// Makefile targets that build for development. A command in one either assigns
/// `RUSTFLAGS` with the standard flags or assigns none and so takes the
/// configuration's. The list is this repository's own, and a target that stops
/// being defined fails the contract rather than dropping out of it.
const DEVELOPMENT_TARGETS: &[&str] = &["test", "typecheck", "lint", "build", "lint-clippy"];
/// Makefile targets that measure or ship, so every command assigns `RUSTFLAGS`
/// and none carries a standard flag.
const HELD_OUT_TARGETS: &[&str] = &["release"];

/// The host `make` is told it runs on, through `BUILD_HOST_OS`.
#[derive(Clone, Copy)]
pub enum Host {
    Linux,
    Darwin,
}

impl Host {
    /// Returns the value `uname -s` reports for the host.
    const fn make_value(self) -> &'static str {
        match self {
            Self::Linux => "Linux",
            Self::Darwin => "Darwin",
        }
    }

    /// Returns whether the host takes mold, which ships for Linux alone.
    pub const fn takes_linker_flag(self) -> bool {
        matches!(self, Self::Linux)
    }
}

/// What one `make -n` command assigns to `RUSTFLAGS`.
#[derive(Debug, PartialEq, Eq)]
pub enum Assignment {
    Unassigned,
    /// An assignment, and whether it keeps the caller's own `RUSTFLAGS`.
    Flags(Flags, bool),
}

/// Reads the `RUSTFLAGS` a `make -n` output line assigns. An unreadable form is
/// an error, because it still replaces the configuration's sources and so must
/// not pass.
///
/// ```text
/// assigned_rustflags("RUSTFLAGS=\"-Zthreads=8\" cargo test") -> Flags(["-Zthreads=8"], inherits: false)
/// assigned_rustflags("RUSTFLAGS=\"${RUSTFLAGS:+$RUSTFLAGS }-Zthreads=8\" cargo test") -> inherits: true
/// assigned_rustflags("cargo test")                           -> Unassigned
/// assigned_rustflags("RUSTFLAGS=-Zthreads=8 cargo test")     -> Err
/// assigned_rustflags("RUSTFLAGS=\"${RUSTFLAGS-}-Zthreads=8\" cargo test") -> Err (glued)
/// ```
///
/// # Errors
///
/// Returns the reason when an assignment is unquoted, unterminated, or glues the
/// caller's flags to the next one.
pub fn assigned_rustflags(line: &str) -> Result<Assignment, String> {
    let Some((_, rest)) = line.split_once("RUSTFLAGS=\"") else {
        if line.contains("RUSTFLAGS=") {
            return Err(format!("unreadable RUSTFLAGS assignment in `{line}`"));
        }
        return Ok(Assignment::Unassigned);
    };
    let (assigned, _) = rest
        .split_once('"')
        .ok_or_else(|| format!("unterminated RUSTFLAGS in `{line}`"))?;
    // The recipes prepend the caller's own flags with these expansions; they are
    // not standard flags. `${RUSTFLAGS-}` adds no separator, so glued to the next
    // word it makes one token with it (`-Dwarnings-Zthreads=8`) and hides the flag.
    let glued = assigned
        .split("${RUSTFLAGS-}")
        .skip(1)
        .any(|after| !after.is_empty() && !after.starts_with(' '));
    if glued {
        return Err(format!(
            "inherited RUSTFLAGS glued to the next flag in `{line}`"
        ));
    }
    let inherits =
        assigned.contains("${RUSTFLAGS:+$RUSTFLAGS }") || assigned.contains("${RUSTFLAGS-}");
    let own = assigned
        .replace("${RUSTFLAGS:+$RUSTFLAGS }", " ")
        .replace("${RUSTFLAGS-}", " ");
    Ok(Assignment::Flags(
        Flags::from_words(own.split_whitespace()),
        inherits,
    ))
}

/// Reads the assignment of each cargo or whitaker command `make -n` printed.
///
/// # Errors
///
/// Returns the reason when a command assigns `RUSTFLAGS` in an unreadable form.
pub fn commands_from(stdout: &str) -> Result<Vec<Assignment>, String> {
    // A recipe continued with a trailing backslash is one command.
    let joined = stdout.replace("\\\n", " ");
    joined
        .lines()
        .filter(|line| !line.trim_start().starts_with("echo"))
        // A tool-availability probe names Cargo but runs no build.
        .filter(|line| !line.trim_start().starts_with("command -v"))
        .filter(|line| line.contains("cargo") || line.contains("whitaker"))
        .map(assigned_rustflags)
        .collect()
}

/// Runs `make -n` for a target on a host and returns what it printed. The tests
/// that drive real `make` use [`real_make`]; a test of the parsing path passes a
/// function that returns canned text instead, so no process runs.
pub type MakeRunner = fn(&str, Host) -> Result<String, String>;

/// The integration adapter: runs the real `make -n` in the crate's directory and
/// reports a spawn failure or an undefined target as an error.
///
/// # Errors
///
/// Returns the reason when `make` cannot run or the target is not defined.
pub fn real_make(target: &str, host: Host) -> Result<String, String> {
    let output = Command::new("make")
        .args([
            "-n",
            "-B",
            &format!("BUILD_HOST_OS={}", host.make_value()),
            target,
        ])
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
        .output()
        .map_err(|error| format!("running make: {error}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!(
            "`make -n {target}` failed, so it is not defined: {stderr}"
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Reads the commands a runner reports for a target on a host.
fn make_commands(runner: MakeRunner, target: &str, host: Host) -> Result<Vec<Assignment>, String> {
    commands_from(&runner(target, host)?)
}

/// Returns the complaint about one development command, if any: an assigned
/// `RUSTFLAGS` keeps the caller's own flags and restates the frontend flag on a
/// nightly pin, and mold on Linux.
pub fn development_problem(
    target: &str,
    host: Host,
    pin: Pin,
    assignment: &Assignment,
) -> Option<String> {
    let Assignment::Flags(flags, inherits) = assignment else {
        return None;
    };
    if !inherits {
        return Some(format!(
            "`make {target}` on {} drops the caller's RUSTFLAGS",
            host.make_value()
        ));
    }
    let reason = flags.meets(pin, host.takes_linker_flag()).err()?;
    Some(format!("`make {target}` on {} {reason}", host.make_value()))
}

/// Returns the complaint when the `test` target keeps `-D warnings` in none of its
/// assigned commands: a recipe may run other commands (a version probe, a
/// prerequisite build) that never carried the policy, but dropping `$(RUST_FLAGS)`
/// from the command that runs the tests drops it from all of them.
pub fn test_policy_problem(target: &str, host: Host, commands: &[Assignment]) -> Option<String> {
    let assigned = commands.iter().filter_map(|command| match command {
        Assignment::Flags(flags, _) => Some(flags),
        Assignment::Unassigned => None,
    });
    let keeps_the_policy = assigned.clone().any(Flags::denies_warnings);
    (target == "test" && assigned.count() > 0 && !keeps_the_policy).then(|| {
        format!(
            "`make {target}` on {} keeps -D warnings in none of its commands",
            host.make_value()
        )
    })
}

/// Returns every complaint about the development targets on one host, and how
/// many assignments it read, so a test can refuse to pass over nothing.
///
/// # Errors
///
/// Returns the reason when a listed target is not defined or unreadable.
pub fn development_problems(
    runner: MakeRunner,
    host: Host,
    pin: Pin,
) -> Result<(Problems, usize), String> {
    let mut problems = Vec::new();
    let mut read = 0;
    for target in DEVELOPMENT_TARGETS {
        let commands = make_commands(runner, target, host)?;
        read += commands
            .iter()
            .filter(|command| **command != Assignment::Unassigned)
            .count();
        problems.extend(
            commands
                .iter()
                .filter_map(|command| development_problem(target, host, pin, command)),
        );
        problems.extend(test_policy_problem(target, host, &commands));
    }
    Ok((problems, read))
}

/// Returns every complaint about one held-out command: it assigns nothing, so
/// it takes the configuration's flags, or the assignment names a standard flag.
fn held_out_command_problems(target: &str, assignment: &Assignment) -> Problems {
    let Assignment::Flags(flags, _) = assignment else {
        return vec![format!(
            "`make {target}` runs a command that takes the configuration's flags"
        )];
    };
    let named = [
        (flags.names_threads(), THREADS_FLAG),
        (flags.names_linker(), LINKER_FLAG),
    ];
    named
        .into_iter()
        .filter(|(is_named, _)| *is_named)
        .map(|(_, flag)| format!("`make {target}` takes {flag}"))
        .collect()
}

/// Returns every complaint about the held-out targets, and how many commands it
/// read: each assigns `RUSTFLAGS`, since only an assignment displaces the
/// configuration's sources.
///
/// # Errors
///
/// Returns the reason when a listed target is not defined or unreadable.
pub fn held_out_problems(runner: MakeRunner) -> Result<(Problems, usize), String> {
    held_out_problems_for(runner, HELD_OUT_TARGETS)
}

/// Returns the complaints about a given list of held-out targets, so a test can name a
/// synthetic target and exercise the check in a repository that defines none.
///
/// # Errors
///
/// Returns the reason when a named target is not defined or unreadable.
pub fn held_out_problems_for(
    runner: MakeRunner,
    targets: &[&str],
) -> Result<(Problems, usize), String> {
    let mut problems = Vec::new();
    let mut read = 0;
    for target in targets {
        let commands = make_commands(runner, target, Host::Linux)?;
        read += commands.len();
        problems.extend(
            commands
                .iter()
                .flat_map(|command| held_out_command_problems(target, command)),
        );
    }
    Ok((problems, read))
}

/// Returns the number of held-out targets the repository defines.
pub const fn held_out_target_count() -> usize {
    HELD_OUT_TARGETS.len()
}
