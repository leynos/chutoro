//! Contract tests for the documentation-denial flags.
//!
//! Chutoro denies missing documentation through `RUSTFLAGS`, not through a
//! `[lints]` table: `-Dmissing_docs` and `-Dmissing_crate_level_docs`. Cargo
//! applies one `rustflags` source rather than merging them, and an assigned
//! `RUSTFLAGS` replaces every source, so the flags are repeated in each Cargo
//! source and restated by each Makefile recipe that assigns `RUSTFLAGS`. The
//! build standard's own contract checks the frontend and linker flags; this one
//! checks that the documentation flags survive the same edits.
//!
//! The Cargo sources are read as text. The recipes are read from what
//! `make -n` prints, on a Linux host, for the development targets and for
//! `release`, which keeps the flags on purpose. One command is exempt: the
//! Whitaker lint run, which applies its own lint set to documentation. The
//! exemption is itself pinned, so it cannot outlive the command it excuses.
//! Fixtures come first, so no rule passes by detecting nothing.

use std::process::Command;

/// The documentation-denial flags every `rustflags` source and recipe carries.
const DOCUMENTATION_FLAGS: [&str; 2] = ["-Dmissing_docs", "-Dmissing_crate_level_docs"];

/// The targets whose recipes assign `RUSTFLAGS` and carry the flags.
const TARGETS: [&str; 5] = ["test", "lint", "typecheck", "build", "release"];

/// The command whose assignment is excused from the documentation flags.
const EXEMPT_COMMAND: &str = "whitaker";

const CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../.cargo/config.toml"
));

/// A `RUSTFLAGS` assignment read from a recipe: its flags, and the command it
/// precedes.
#[derive(Debug, PartialEq, Eq)]
struct Assignment {
    flags: Vec<String>,
    command: String,
}

/// Returns the flags a `rustflags = [...]` line lists, or `None` when the line
/// is not a `rustflags` entry.
///
/// An entry whose array does not close on its own line is an error: reading
/// half of it would let a lost flag pass.
fn listed_flags(line: &str) -> Option<Result<Vec<String>, String>> {
    let (key, value) = line.split_once('=')?;
    if key.trim() != "rustflags" {
        return None;
    }
    let Some(inner) = value
        .trim()
        .strip_prefix('[')
        .and_then(|rest| rest.split_once(']'))
        .map(|(list, _)| list)
    else {
        return Some(Err(format!(
            "`{line}`: the array does not close on its line"
        )));
    };
    Some(Ok(inner
        .split(',')
        .map(|item| item.trim().trim_matches('"').to_owned())
        .filter(|item| !item.is_empty())
        .collect()))
}

/// Returns the complaints about a Cargo configuration: each `rustflags` source
/// must list every documentation flag, and there must be a source to read. A
/// commented-out entry is no source, because its key reads as `# rustflags`.
fn config_problems(config: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let mut sources = 0_usize;
    for line in config.lines().map(str::trim) {
        match listed_flags(line) {
            None => {}
            Some(Err(reason)) => problems.push(reason),
            Some(Ok(flags)) => {
                sources += 1;
                problems.extend(
                    DOCUMENTATION_FLAGS
                        .iter()
                        .filter(|wanted| !flags.iter().any(|flag| flag == *wanted))
                        .map(|wanted| format!("a rustflags source {flags:?} lacks {wanted}")),
                );
            }
        }
    }
    if sources == 0 {
        problems.push("no rustflags source was found, so the check proves nothing".to_owned());
    }
    problems
}

/// Returns the `RUSTFLAGS` assignments that precede a command in `make -n`
/// output, skipping lines that assign none.
///
/// ```text
/// RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" cargo test
///   -> Assignment { flags: ["-D", "warnings"], command: "cargo" }
/// ```
fn assignments_in(output: &str) -> Vec<Assignment> {
    output
        .lines()
        .filter_map(|line| {
            let rest = line.split_once("RUSTFLAGS=\"")?.1;
            let (value, after) = rest.split_once('"')?;
            let flags = value
                .replace("${RUSTFLAGS:+$RUSTFLAGS }", "")
                .split_whitespace()
                .map(str::to_owned)
                .collect();
            let command = after.split_whitespace().next()?.to_owned();
            Some(Assignment { flags, command })
        })
        .collect()
}

/// Returns the complaints about one target's assignments: each must list every
/// documentation flag, unless its command is the exempt one.
fn assignment_problems(target: &str, assignments: &[Assignment]) -> Vec<String> {
    let mut problems = Vec::new();
    if assignments.is_empty() {
        problems.push(format!(
            "`make {target}` assigns no RUSTFLAGS, so the check proves nothing"
        ));
    }
    for assignment in assignments
        .iter()
        .filter(|assignment| !assignment.command.ends_with(EXEMPT_COMMAND))
    {
        problems.extend(
            DOCUMENTATION_FLAGS
                .iter()
                .filter(|wanted| !assignment.flags.iter().any(|flag| flag == *wanted))
                .map(|wanted| {
                    format!(
                        "`make {target}`: `{}` runs without {wanted}: {:?}",
                        assignment.command, assignment.flags
                    )
                }),
        );
    }
    problems
}

/// Returns what `make -n` prints for a target on a Linux host.
fn make_output(target: &str) -> Result<String, String> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let output = Command::new("make")
        .args(["-n", "-B", target, "BUILD_HOST_OS=Linux"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("could not run make: {error}"))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(format!(
            "`make -n {target}` failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Turns a list of complaints into a test result.
fn none_of(problems: &[String]) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{problems:#?}"))
    }
}

const BOTH_SOURCES: &str = r#"
[build]
rustflags = ["-Dmissing_docs", "-Dmissing_crate_level_docs"]

[target.'cfg(target_os = "linux")']
rustflags = ["-Dmissing_docs", "-Dmissing_crate_level_docs", "-Clink-arg=-fuse-ld=mold"]
"#;

const BUILD_LOSES_ONE: &str = r#"
[build]
rustflags = ["-Dmissing_docs"]

[target.'cfg(target_os = "linux")']
rustflags = ["-Dmissing_docs", "-Dmissing_crate_level_docs", "-Clink-arg=-fuse-ld=mold"]
"#;

const LINUX_LOSES_BOTH: &str = r#"
[build]
rustflags = ["-Dmissing_docs", "-Dmissing_crate_level_docs"]

[target.'cfg(target_os = "linux")']
rustflags = ["-Clink-arg=-fuse-ld=mold"]
"#;

const COMMENTED_FLAGS: &str = r#"
[build]
# rustflags = ["-Dmissing_docs", "-Dmissing_crate_level_docs"]
rustflags = ["-Dmissing_docs"]
"#;

const SPREAD_ARRAY: &str = r#"
[build]
rustflags = [
    "-Dmissing_docs",
    "-Dmissing_crate_level_docs",
]
"#;

const LOOKALIKE_KEY: &str = r#"
[build]
rustflags_extra = ["-Dmissing_docs", "-Dmissing_crate_level_docs"]
"#;

#[test]
fn the_cargo_configuration_carries_the_documentation_flags() -> Result<(), String> {
    none_of(&config_problems(CONFIG))
}

#[test]
fn a_source_with_both_flags_is_accepted() -> Result<(), String> {
    none_of(&config_problems(BOTH_SOURCES))
}

#[test]
fn a_source_that_loses_a_flag_is_refused() {
    assert_eq!(config_problems(BUILD_LOSES_ONE).len(), 1);
    assert_eq!(config_problems(LINUX_LOSES_BOTH).len(), 2);
    assert_eq!(config_problems(COMMENTED_FLAGS).len(), 1);
}

#[test]
fn a_configuration_with_no_source_proves_nothing() {
    assert_eq!(config_problems("").len(), 1);
    assert_eq!(config_problems(LOOKALIKE_KEY).len(), 1);
}

#[test]
fn a_rustflags_array_spread_over_lines_is_refused() {
    assert!(!config_problems(SPREAD_ARRAY).is_empty());
}

#[test]
fn an_assignment_is_read_with_the_inherited_value_set_aside() {
    let read = assignments_in(
        "RUSTFLAGS=\"${RUSTFLAGS:+$RUSTFLAGS }-D warnings -Dmissing_docs\" cargo test\necho done\n",
    );
    assert_eq!(
        read,
        [Assignment {
            flags: ["-D", "warnings", "-Dmissing_docs"]
                .map(str::to_owned)
                .to_vec(),
            command: "cargo".to_owned(),
        }]
    );
}

#[test]
fn an_assignment_that_loses_a_flag_is_refused_unless_it_is_the_exempt_command() {
    let lacking = assignments_in("RUSTFLAGS=\"-Dmissing_docs\" cargo clippy\n");
    assert_eq!(assignment_problems("lint", &lacking).len(), 1);
    let whitaker = assignments_in("RUSTFLAGS=\"-D warnings\" whitaker --all\n");
    assert!(assignment_problems("lint", &whitaker).is_empty());
    let complete =
        assignments_in("RUSTFLAGS=\"-Dmissing_docs -Dmissing_crate_level_docs\" cargo clippy\n");
    assert!(assignment_problems("lint", &complete).is_empty());
}

#[test]
fn a_target_that_assigns_nothing_proves_nothing() {
    assert_eq!(
        assignment_problems("lint", &assignments_in("cargo clippy\n")).len(),
        1
    );
}

#[test]
fn every_recipe_that_assigns_rustflags_restates_the_documentation_flags() -> Result<(), String> {
    for target in TARGETS {
        let assignments = assignments_in(&make_output(target)?);
        none_of(&assignment_problems(target, &assignments))?;
    }
    Ok(())
}

#[test]
fn the_exemption_is_held_by_a_command_that_still_exists() -> Result<(), String> {
    let exempt = assignments_in(&make_output("lint")?)
        .iter()
        .filter(|assignment| assignment.command.ends_with(EXEMPT_COMMAND))
        .count();
    if exempt == 1 {
        Ok(())
    } else {
        Err(format!(
            "`make lint` should run `{EXEMPT_COMMAND}` once under an assignment, found {exempt}; \
             remove the exemption if the command is gone"
        ))
    }
}
