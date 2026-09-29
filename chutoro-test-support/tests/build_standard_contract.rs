//! Contract tests for the Rust build standard.
//!
//! The standard makes the parallel `rustc` frontend the default for every
//! development build on a nightly pin, and mold the default linker on Linux.
//! Cargo reads both from `.cargo/config.toml`, but it applies a single
//! `rustflags` source rather than merging them, and an assigned `RUSTFLAGS`
//! replaces every source. So the flags must be repeated in each configuration
//! source, restated wherever the Makefile assigns `RUSTFLAGS` for a development
//! target, and kept out of the coverage and release recipes, which measure or
//! ship and so stay on the default flags. A stable pin takes mold alone,
//! because `-Zthreads` is a nightly flag.
//!
//! The Makefile clauses run `make -n` and read the commands it would run,
//! rather than the Makefile's text, so a flag lost through a variable or a
//! recipe edit fails here. They run once as a Linux host and once as a macOS
//! host through `BUILD_HOST_OS`, because mold is added on Linux alone. The
//! listed targets and workflows are this repository's own: one that stops being
//! defined fails the contract, so the check cannot quietly stop covering it.
//! Each CI workflow that sets up Rust passes `install-mold: 'true'`, so the
//! Linux jobs have the linker. The readers are driven against fixtures first,
//! because a rule exercised only over this repository's own compliant files
//! would pass whether or not it detects anything.

#[path = "build_standard_support/ci_steps.rs"]
mod ci_steps;
#[path = "build_standard_support/config.rs"]
mod config;
#[path = "build_standard_support/make.rs"]
mod make;
use rstest::rstest;

use ci_steps::{coverage_problems, linker_install_problems, workflow_problems};
use config::{CONFIG, Flags, Pin, Problems, THREADS_FLAG, TOOLCHAIN, config_problems};
use make::{
    Assignment, Host, assigned_rustflags, commands_from, development_problems, held_out_problems,
    held_out_target_count,
};

/// Turns a list of complaints into a test result.
fn none_of(problems: &Problems) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{problems:#?}"))
    }
}

/// A toolchain file pinning a nightly channel.
const NIGHTLY: &str = "[toolchain]\nchannel = \"nightly-2026-05-28\"\n";
/// A toolchain file pinning a stable channel.
const STABLE: &str = "[toolchain]\nchannel = \"1.94.0\"\n";

/// A compliant nightly configuration: the frontend flag in every source and
/// mold in the Linux table alone.
const NIGHTLY_OK: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\n",
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// The same, with the linker flag spelled as the `-C` pair Cargo also accepts.
const NIGHTLY_SPELLED_APART: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\n",
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\", \"-C\", \"link-arg=-fuse-ld=mold\"]\n"
);
/// A compliant stable configuration: mold alone, in the Linux table.
const STABLE_OK: &str = concat!(
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A nightly configuration whose `[build]` source lost the frontend flag, so it
/// is missing it and also differs from the Linux source.
const BUILD_LOSES_THREADS: &str = concat!(
    "[build]\nrustflags = [\"-Dwarnings\"]\n",
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A nightly configuration whose Linux table lost mold.
const LINUX_LOSES_LINKER: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\"]\n",
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\"]\n"
);
/// A nightly configuration that names mold in `[build]`, beyond Linux.
const LINKER_IN_BUILD: &str = concat!(
    "[build]\nrustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n",
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A nightly configuration with no `[build]` source for the other hosts.
const NO_BUILD_SOURCE: &str = concat!(
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A stable configuration that names the nightly-only frontend flag.
const STABLE_WITH_THREADS: &str = concat!(
    "[target.x86_64-unknown-linux-gnu]\nlinker = \"clang\"\n",
    "rustflags = [\"-Zthreads=8\", \"-Clink-arg=-fuse-ld=mold\"]\n"
);
/// A `rustflags` array spread over several lines, which the reader refuses.
const SPREAD_ARRAY: &str = "[build]\nrustflags = [\n  \"-Zthreads=8\",\n]\n";

/// Checks that a fixture configuration draws the expected number of complaints.
fn draws(config: &str, pin: Pin, expected: usize) -> Result<(), String> {
    let found = config_problems(config, pin)?.len();
    if found == expected {
        Ok(())
    } else {
        Err(format!("{config:?}: {found} problems, not {expected}"))
    }
}

/// Scenario: configurations of each shape, read on a nightly and a stable pin.
///
/// Invariant: a compliant nightly file passes, and each way of losing the
/// frontend flag, losing mold, naming mold beyond Linux, or letting a source
/// drift is reported; a stable pin refuses the frontend flag it cannot take.
#[rstest]
#[case::compliant_nightly(NIGHTLY_OK, Pin::Nightly, 0)]
#[case::linker_spelled_as_a_pair(NIGHTLY_SPELLED_APART, Pin::Nightly, 0)]
#[case::compliant_stable(STABLE_OK, Pin::Stable, 0)]
#[case::build_loses_the_frontend(BUILD_LOSES_THREADS, Pin::Nightly, 2)]
#[case::linux_loses_the_linker(LINUX_LOSES_LINKER, Pin::Nightly, 1)]
#[case::linker_named_in_build(LINKER_IN_BUILD, Pin::Nightly, 1)]
#[case::no_build_source(NO_BUILD_SOURCE, Pin::Nightly, 1)]
#[case::stable_names_the_frontend(STABLE_WITH_THREADS, Pin::Stable, 1)]
#[case::empty_configuration("", Pin::Nightly, 3)]
fn the_configuration_reader_reports_each_defect(
    #[case] config: &str,
    #[case] pin: Pin,
    #[case] expected: usize,
) -> Result<(), String> {
    draws(config, pin, expected)
}

/// Scenario: a `rustflags` array spread over several lines.
///
/// Invariant: the reader refuses it, because reading half of an entry would let
/// a lost flag pass.
#[test]
fn a_rustflags_array_spread_over_lines_is_refused() -> Result<(), String> {
    match config_problems(SPREAD_ARRAY, Pin::Nightly) {
        Ok(_) => Err("a rustflags array spread over lines was read".to_owned()),
        Err(_) => Ok(()),
    }
}

/// A toolchain file that names no channel.
const NO_CHANNEL: &str = "[toolchain]\ncomponents = [\"clippy\"]\n";
/// A toolchain file that names two channels.
const TWO_CHANNELS: &str = "[toolchain]\nchannel = \"stable\"\nchannel = \"nightly\"\n";
/// A toolchain file naming a channel the standard does not know.
const UNKNOWN_CHANNEL: &str = "[toolchain]\nchannel = \"weekly\"\n";

/// Scenario: toolchain files pinning each kind of channel, and files that do
/// not.
///
/// Invariant: only a `nightly` channel reads as nightly, so only it is asked to
/// carry `-Zthreads`; a missing, repeated or unknown channel is an error, not a
/// stable pin by default.
#[rstest]
#[case::nightly(NIGHTLY, Some(Pin::Nightly))]
#[case::stable(STABLE, Some(Pin::Stable))]
#[case::missing(NO_CHANNEL, None)]
#[case::repeated(TWO_CHANNELS, None)]
#[case::unknown(UNKNOWN_CHANNEL, None)]
fn the_pin_reader_tells_the_channels_apart(#[case] toolchain: &str, #[case] expected: Option<Pin>) {
    assert_eq!(Pin::read(toolchain).ok(), expected);
}

/// Builds the assignment a fixture line is expected to read as.
fn flags(words: &[&str], inherits: bool) -> Assignment {
    Assignment::Flags(Flags::from_words(words.iter().copied()), inherits)
}

/// Scenario: `make -n` output lines in each spelling of an assignment.
///
/// Invariant: a quoted assignment is read, with the caller's inherited flags
/// set aside, and a line assigning none reads as unassigned.
#[rstest]
#[case::plain("RUSTFLAGS=\"-D warnings -Zthreads=8\" cargo test", flags(&["-D", "warnings", THREADS_FLAG], false))]
#[case::inherited_flags_glued_on(
    "RUSTFLAGS=\"${RUSTFLAGS:+$RUSTFLAGS }-Zthreads=8\" cargo check",
    flags(&[THREADS_FLAG], true)
)]
#[case::inherited_flags_only("RUSTFLAGS=\"${RUSTFLAGS-}\" cargo build --release", flags(&[], true))]
#[case::no_assignment("cargo clippy --all-targets", Assignment::Unassigned)]
fn the_command_reader_reads_each_assignment(
    #[case] line: &str,
    #[case] expected: Assignment,
) -> Result<(), String> {
    if assigned_rustflags(line)? == expected {
        Ok(())
    } else {
        Err(format!("`{line}` was read wrongly"))
    }
}

/// Scenario: `make -n` output lines whose assignment the reader cannot parse.
///
/// Invariant: each is refused rather than passed, because an assignment in a
/// form the reader does not understand still replaces the configuration.
#[rstest]
#[case::unquoted("RUSTFLAGS=-Zthreads=8 cargo test")]
#[case::unterminated("RUSTFLAGS=\"-Zthreads=8 cargo test")]
fn the_command_reader_refuses_what_it_cannot_parse(#[case] line: &str) -> Result<(), String> {
    match assigned_rustflags(line) {
        Ok(_) => Err(format!("`{line}` was read, not refused")),
        Err(_) => Ok(()),
    }
}

/// A workflow step that passes the input, quoted.
const STEP_INSTALLS: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "        with:\n          install-mold: 'true'\n"
);
/// The same, with the bare value.
const STEP_INSTALLS_BARE: &str = concat!(
    "    steps:\n      - uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "        with:\n          install-mold: true\n"
);
/// A step with no input at all.
const STEP_MISSING_INPUT: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n"
);
/// A step that turns the input off.
const STEP_INPUT_OFF: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "        with:\n          install-mold: 'false'\n"
);
/// A step without the input, followed by a step that has one for another
/// action.
const STEP_BEFORE_A_SIBLING_THAT_INSTALLS: &str = concat!(
    "    steps:\n      - name: Setup Rust\n",
    "        uses: org/shared-actions/.github/actions/setup-rust@0123456789abcdef0123456789abcdef01234567\n",
    "      - name: Other\n        uses: org/other@abc\n        with:\n          install-mold: 'true'\n"
);
/// A comment that names the action, and no step.
const COMMENT_NAMING_THE_ACTION: &str =
    "    steps:\n      # setup-rust@abc installs it\n      - run: make\n";

/// Scenario: workflow steps that set up Rust with and without the input.
///
/// Invariant: a step must pass `install-mold: 'true'` itself; another step's
/// input does not count, and a comment naming the action is not a step.
#[rstest]
#[case::quoted_true(STEP_INSTALLS, 0)]
#[case::bare_true(STEP_INSTALLS_BARE, 0)]
#[case::missing_input(STEP_MISSING_INPUT, 1)]
#[case::input_off(STEP_INPUT_OFF, 1)]
#[case::input_on_a_sibling_step(STEP_BEFORE_A_SIBLING_THAT_INSTALLS, 1)]
#[case::comment_only(COMMENT_NAMING_THE_ACTION, 0)]
fn the_workflow_reader_wants_the_input_on_each_step(
    #[case] workflow: &str,
    #[case] expected: usize,
) -> Result<(), String> {
    let found = linker_install_problems("fixture.yml", workflow).len();
    if found == expected {
        Ok(())
    } else {
        Err(format!("{workflow:?}: {found} problems, not {expected}"))
    }
}

/// A coverage step that assigns `RUSTFLAGS` without a standard flag.
const COVERAGE_OK: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "        env:\n          RUSTFLAGS: -D warnings\n"
);
/// A coverage step with no assignment.
const COVERAGE_UNASSIGNED: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n"
);
/// A coverage step that takes the frontend flag.
const COVERAGE_WITH_THREADS: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "        env:\n          RUSTFLAGS: -D warnings -Zthreads=8\n"
);
/// A coverage step that takes mold.
const COVERAGE_WITH_LINKER: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "        env:\n          RUSTFLAGS: -Clink-arg=-fuse-ld=mold\n"
);
/// A coverage step whose assignment belongs to the next step.
const COVERAGE_BORROWING_A_SIBLING: &str = concat!(
    "    steps:\n      - name: Cover\n",
    "        uses: org/shared-actions/.github/actions/generate-coverage@0123456789abcdef0123456789abcdef01234567\n",
    "      - name: Other\n        env:\n          RUSTFLAGS: -D warnings\n"
);

/// Scenario: coverage steps with and without an explicit assignment.
///
/// Invariant: the step assigns `RUSTFLAGS` itself and names neither standard
/// flag; a sibling step's assignment does not count.
#[rstest]
#[case::assigned(COVERAGE_OK, 0)]
#[case::unassigned(COVERAGE_UNASSIGNED, 1)]
#[case::with_the_frontend_flag(COVERAGE_WITH_THREADS, 1)]
#[case::with_the_linker(COVERAGE_WITH_LINKER, 1)]
#[case::assignment_on_a_sibling_step(COVERAGE_BORROWING_A_SIBLING, 1)]
fn the_coverage_reader_wants_an_explicit_assignment(
    #[case] workflow: &str,
    #[case] expected: usize,
) -> Result<(), String> {
    let found = coverage_problems("fixture.yml", workflow).len();
    if found == expected {
        Ok(())
    } else {
        Err(format!("{workflow:?}: {found} problems, not {expected}"))
    }
}

/// Every workflow that builds under the standard installs mold. A repository
/// whose workflows do not set up Rust through `setup-rust` lists none, and the
/// check then reads nothing; a listed workflow must have a step to read.
#[test]
fn every_setup_rust_step_installs_linker() -> Result<(), String> {
    none_of(&workflow_problems())
}

/// Scenario: a recipe continued over lines, beside an `echo` and another
/// command.
///
/// Invariant: the continued command is one command, and lines that are not a
/// Cargo or Whitaker command are ignored.
#[test]
fn a_continued_command_is_one_command() -> Result<(), String> {
    let joined = commands_from(concat!(
        "RUSTFLAGS=\"-A\" \\\n",
        "cargo test\necho cargo test\nmake other\n"
    ))?;
    if joined == vec![flags(&["-A"], false)] {
        Ok(())
    } else {
        Err(format!("read wrongly: {joined:?}"))
    }
}

#[test]
fn every_rustflags_source_is_consistent_with_the_pin() -> Result<(), String> {
    none_of(&config_problems(CONFIG, Pin::read(TOOLCHAIN)?)?)
}

#[test]
fn development_targets_restate_the_flags_on_linux() -> Result<(), String> {
    let (problems, read) = development_problems(Host::Linux, Pin::read(TOOLCHAIN)?)?;
    none_of(&problems)?;
    if read == 0 {
        return Err(
            "no development target assigns RUSTFLAGS, so the check proves nothing".to_owned(),
        );
    }
    Ok(())
}

#[test]
fn development_targets_keep_the_frontend_but_not_the_linker_elsewhere() -> Result<(), String> {
    none_of(&development_problems(Host::Darwin, Pin::read(TOOLCHAIN)?)?.0)
}

/// Coverage measures and release ships, so both stay on the default flags. A
/// repository that lists no such target has nothing local to hold out, and the
/// check then reads no commands; otherwise it must read at least one.
#[test]
fn coverage_and_release_take_neither_flag() -> Result<(), String> {
    let (problems, read) = held_out_problems()?;
    none_of(&problems)?;
    if held_out_target_count() > 0 && read == 0 {
        return Err(
            "the held-out targets run no cargo command, so the check proves nothing".to_owned(),
        );
    }
    Ok(())
}
