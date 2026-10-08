//! Reader cases for the commands `make -n` prints: which a recipe must give `RUSTFLAGS`, and how a
//! line that chains several commands is read one command at a time.

use rstest::rstest;

use super::make::{Assignment, commands_from};

/// Scenario: `make -n` output for commands that assign no `RUSTFLAGS`.
///
/// Invariant: a build, test or lint command that assigns nothing is read as bare, which a
/// development recipe must not leave; a formatter, a metadata probe and a documentation
/// build stay unassigned, because they run no compiled code under test.
#[rstest]
#[case::cargo_test("cargo test\n", Assignment::Bare("cargo test".to_owned()))]
#[case::cargo_by_path("/home/user/.cargo/bin/cargo test --workspace\n", Assignment::Bare("/home/user/.cargo/bin/cargo test --workspace".to_owned()))]
#[case::nextest("cargo +nightly nextest run\n", Assignment::Bare("cargo +nightly nextest run".to_owned()))]
#[case::clippy("cargo clippy --all-targets\n", Assignment::Bare("cargo clippy --all-targets".to_owned()))]
#[case::typecheck("cargo check --workspace\n", Assignment::Bare("cargo check --workspace".to_owned()))]
#[case::whitaker("whitaker --all\n", Assignment::Bare("whitaker --all".to_owned()))]
#[case::llvm_cov("cargo llvm-cov nextest\n", Assignment::Bare("cargo llvm-cov nextest".to_owned()))]
#[case::version_probe("cargo nextest --version\n", Assignment::Unassigned)]
#[case::formatter("cargo fmt --all --check\n", Assignment::Unassigned)]
#[case::metadata("cargo metadata --format-version 1\n", Assignment::Unassigned)]
#[case::documentation(
    "RUSTDOCFLAGS=\"-D warnings\" cargo doc --workspace\n",
    Assignment::Unassigned
)]
fn the_command_reader_tells_a_bare_tool_from_an_exempt_command(
    #[case] stdout: &str,
    #[case] expected: Assignment,
) -> Result<(), String> {
    let read = commands_from(stdout)?;
    if read == [expected] {
        Ok(())
    } else {
        Err(format!("`{stdout}` was read as {read:?}"))
    }
}

/// Counts the assigned and the bare commands the reader finds in `make -n` output.
fn tally(stdout: &str) -> Result<(usize, usize), String> {
    let read = commands_from(stdout)?;
    let assigned = read
        .iter()
        .filter(|command| matches!(command, Assignment::Flags(..)))
        .count();
    let bare = read
        .iter()
        .filter(|command| matches!(command, Assignment::Bare(..)))
        .count();
    Ok((assigned, bare))
}

/// Scenario: `make -n` lines that chain several commands with `;`, `&&`, `||` and shell keywords.
///
/// Invariant: each command is judged on its own, so a command that assigns nothing is found
/// beside a probe or an inspection command that is exempt or assigns, a separator inside quotes
/// splits nothing, and an `echo` of a message that names Cargo is not a command.
#[rstest]
#[case::probe_run_and_fallback_all_assign(
    "if RUSTFLAGS=\"A\" cargo nextest --version >/dev/null 2>&1; then RUSTFLAGS=\"A\" cargo nextest run; else echo \"falling back to cargo test\"; RUSTFLAGS=\"A\" cargo test; fi\n",
    (3, 0)
)]
#[case::the_real_command_is_bare_beside_a_probe(
    "if RUSTFLAGS=\"A\" cargo nextest --version; then cargo nextest run; else RUSTFLAGS=\"A\" cargo test; fi\n",
    (2, 1)
)]
#[case::a_bare_build_after_metadata("cargo metadata --format-version 1 && cargo build --locked\n", (0, 1))]
#[case::an_assigned_build_after_metadata("cargo metadata --format-version 1 && RUSTFLAGS=\"A\" cargo build\n", (1, 0))]
#[case::a_bare_build_after_or("true || cargo build\n", (0, 1))]
#[case::a_separator_inside_quotes("RUSTFLAGS=\"A\" cargo test -- --skip \"a;b&&c\"\n", (1, 0))]
#[case::a_pipe_is_not_a_separator("cargo metadata --format-version 1 | jq .\n", (0, 0))]
fn the_command_reader_judges_each_chained_command(
    #[case] stdout: &str,
    #[case] expected: (usize, usize),
) -> Result<(), String> {
    let found = tally(stdout)?;
    if found == expected {
        Ok(())
    } else {
        Err(format!("`{stdout}`: read {found:?}, expected {expected:?}"))
    }
}
