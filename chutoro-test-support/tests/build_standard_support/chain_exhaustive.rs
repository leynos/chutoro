//! Bounded exhaustive test for the command reader's handling of chained shell lines: every
//! small line drawn from a few command shapes is rendered, and the reader's verdict is checked
//! against what the generator put on it.

use super::{
    exhaustive::sequences,
    make::{Assignment, commands_from},
};

/// A command the chained-line test can place on a line, and how many assigned and bare builds it is.
#[derive(Clone, Copy)]
struct ChainedCommand {
    text: &'static str,
    assigned: usize,
    bare: usize,
}

const CHAINED_COMMANDS: [ChainedCommand; 5] = [
    ChainedCommand {
        text: "RUSTFLAGS=\"-D warnings\" cargo test",
        assigned: 1,
        bare: 0,
    },
    ChainedCommand {
        text: "cargo build --locked",
        assigned: 0,
        bare: 1,
    },
    ChainedCommand {
        text: "cargo metadata --format-version 1",
        assigned: 0,
        bare: 0,
    },
    ChainedCommand {
        text: "cargo nextest --version >/dev/null 2>&1",
        assigned: 0,
        bare: 0,
    },
    ChainedCommand {
        text: "echo \"cargo test; cargo build && cargo clippy\"",
        assigned: 0,
        bare: 0,
    },
];

/// Counts the assigned and the bare commands the reader finds on a line.
fn found_on(line: &str) -> Result<(usize, usize), String> {
    let read = commands_from(line)?;
    let count =
        |wanted: fn(&Assignment) -> bool| read.iter().filter(|command| wanted(command)).count();
    Ok((
        count(|command| matches!(command, Assignment::Flags(..))),
        count(|command| matches!(command, Assignment::Bare(..))),
    ))
}

/// Checks the reader's verdict on one generated line against what the generator put on it.
fn check_chained_line(
    lead: &str,
    separator: &str,
    chosen: &[ChainedCommand],
) -> Result<(), String> {
    let texts: Vec<&str> = chosen.iter().map(|command| command.text).collect();
    let line = format!("{lead}{}\n", texts.join(separator));
    let expected = (
        chosen.iter().map(|command| command.assigned).sum::<usize>(),
        chosen.iter().map(|command| command.bare).sum::<usize>(),
    );
    let found = found_on(&line)?;
    if found == expected {
        Ok(())
    } else {
        Err(format!("`{line}`: read {found:?}, expected {expected:?}"))
    }
}

/// Scenario: every `make -n` line of up to three commands drawn from an assigned build, a bare
/// build, an inspection command, a version probe and an `echo` that names Cargo, joined by each
/// separator and optionally led by `if`.
///
/// Invariant: the reader finds exactly one assigned command per assigned build and one bare
/// command per bare build, whatever sits beside it: a separator splits, a quoted separator does
/// not, and neither an inspection command, a probe nor an `echo` is ever read as a build.
#[test]
fn the_command_reader_finds_each_build_in_every_chained_line() -> Result<(), String> {
    let sequences_of_commands = sequences(&CHAINED_COMMANDS, 3);
    for separator in ["; ", " && ", " || ", "; then ", "; else ", "; do "] {
        for lead in ["", "if "] {
            for chosen in sequences_of_commands
                .iter()
                .filter(|chosen| !chosen.is_empty())
            {
                check_chained_line(lead, separator, chosen)?;
            }
        }
    }
    Ok(())
}
