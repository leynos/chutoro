//! Check one Kani run against the checked-in harness inventory.
//!
//! Usage:
//!
//! ```plaintext
//! kani_suite --manifest tools/kani/harnesses.toml --package PACKAGE \
//!     --scope fast|full -- [NAME=VALUE...] COMMAND [ARGUMENT...]
//! ```
//!
//! The command after `--` is the `cargo kani` invocation the Makefile would
//! have run. The checker refuses one that suppresses or redirects its output,
//! runs it, and fails unless the run completed exactly the manifest's
//! harnesses for that package and tier.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use chutoro_test_support::ci::kani_suite::manifest::Tier;
use chutoro_test_support::ci::kani_suite::{Request, check};

/// The exit status used for a bad invocation, which is not a proof verdict.
const USAGE_FAILURE: u8 = 2;

/// The usage banner shown for a malformed invocation.
const USAGE: &str = concat!(
    "usage: kani_suite --manifest PATH --package PACKAGE ",
    "--scope fast|full -- [NAME=VALUE...] COMMAND [ARGUMENT...]"
);

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match parse(&arguments) {
        Ok(request) => report(&request),
        Err(message) => usage(&message),
    }
}

/// Report a malformed invocation and return its status.
fn usage(message: &str) -> ExitCode {
    let mut stderr = std::io::stderr().lock();
    write_line(&mut stderr, &format!("kani_suite: {message}"));
    write_line(&mut stderr, USAGE);
    ExitCode::from(USAGE_FAILURE)
}

/// Run one parsed request and report its outcome as a process status.
fn report(request: &Request) -> ExitCode {
    match check(request) {
        Ok(verified) => {
            write_line(
                &mut std::io::stdout().lock(),
                &format!(
                    "kani_suite: verified {} harness(es) for {} ({})",
                    verified.harnesses.len(),
                    verified.package,
                    verified.scope
                ),
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            write_line(
                &mut std::io::stderr().lock(),
                &format!("kani_suite: {error}"),
            );
            ExitCode::FAILURE
        }
    }
}

/// Parse the command line into a request, or return a usage message.
fn parse(arguments: &[String]) -> Result<Request, String> {
    let mut manifest: Option<PathBuf> = None;
    let mut package: Option<String> = None;
    let mut scope: Option<Tier> = None;
    let mut command: Option<Vec<String>> = None;
    let mut index = 0;
    while index < arguments.len() {
        let Some(argument) = arguments.get(index) else {
            break;
        };
        if argument == "--" {
            command = Some(arguments.get(index + 1..).unwrap_or_default().to_vec());
            break;
        }
        let (name, inline) = split_option(argument);
        match name {
            "--manifest" => {
                manifest = Some(PathBuf::from(value(arguments, &mut index, name, inline)?));
            }
            "--package" => package = Some(value(arguments, &mut index, name, inline)?),
            "--scope" => {
                scope = Some(scope_of(&value(arguments, &mut index, name, inline)?)?);
            }
            "--help" | "-h" => return Err(String::from("help requested")),
            other => return Err(format!("unexpected argument {other}")),
        }
        index += 1;
    }
    Ok(Request {
        manifest: manifest.ok_or("--manifest is required")?,
        package: package.ok_or("--package is required")?,
        scope: scope.ok_or("--scope is required")?,
        command: command.ok_or("the command after '--' is required")?,
    })
}

/// Split `--name=value` into its option name and inline value.
fn split_option(argument: &str) -> (&str, Option<String>) {
    match argument.split_once('=') {
        Some((name, value)) => (name, Some(value.to_owned())),
        None => (argument, None),
    }
}

/// Return one option's value, reading the next argument when none is inline.
fn value(
    arguments: &[String],
    index: &mut usize,
    name: &str,
    inline: Option<String>,
) -> Result<String, String> {
    if let Some(text) = inline {
        return Ok(text);
    }
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| format!("{name} requires a value"))
}

/// Parse the `--scope` value.
fn scope_of(value: &str) -> Result<Tier, String> {
    match value {
        "fast" => Ok(Tier::Fast),
        "full" => Ok(Tier::Full),
        other => Err(format!(
            "unknown scope {other:?}; expected \"fast\" or \"full\""
        )),
    }
}

/// Write one line, ignoring a closed stream.
fn write_line(sink: &mut impl Write, line: &str) {
    if writeln!(sink, "{line}").is_err() {
        // A closed stream cannot change the verdict.
    }
}
