//! Contract tests for the Rust build standard.
//!
//! The standard makes the parallel `rustc` frontend and, on Linux, mold the
//! default for every development build. Cargo applies one `rustflags` source
//! rather than merging them, and an assigned `RUSTFLAGS` replaces every source,
//! so the flags are repeated in each configuration source, restated by each
//! development recipe, and kept out of the coverage and release recipes. The
//! Makefile clauses read what `make -n` prints on a Linux and a macOS host; each
//! listed `setup-rust` step passes `install-mold`. Fixtures come first, so no
//! rule passes by detecting nothing.

#[path = "build_standard_support/chain_exhaustive.rs"]
mod chain_exhaustive;
#[path = "build_standard_support/ci_steps.rs"]
mod ci_steps;
#[path = "build_standard_support/command_reader.rs"]
mod command_reader;
#[path = "build_standard_support/config.rs"]
mod config;
#[path = "build_standard_support/exhaustive.rs"]
mod exhaustive;
#[path = "build_standard_support/fixtures.rs"]
mod fixtures;
#[path = "build_standard_support/injected.rs"]
mod injected;
#[path = "build_standard_support/make.rs"]
mod make;
#[path = "build_standard_support/shell.rs"]
mod shell;
#[path = "build_standard_support/workflow_exhaustive.rs"]
mod workflow_exhaustive;
use rstest::rstest;

use fixtures::{
    BUILD_LOSES_THREADS, COMMENT_NAMING_THE_ACTION, COMMENTED_OK, COVERAGE_BORROWING_A_SIBLING,
    COVERAGE_COMMENTED_POLICY, COVERAGE_DENYING_WITH_COMMENT, COVERAGE_EMPTY_POLICY,
    COVERAGE_LOOKALIKE_POLICY, COVERAGE_OK, COVERAGE_OTHER_POLICY, COVERAGE_UNASSIGNED,
    COVERAGE_WITH_LINKER, COVERAGE_WITH_THREADS, LINKER_IN_BUILD, LINUX_LOSES_LINKER, NIGHTLY,
    NIGHTLY_OK, NIGHTLY_SPELLED_APART, NO_BUILD_SOURCE, NO_CHANNEL, SIBLING_KEY_OK, SPREAD_ARRAY,
    STABLE, STABLE_OK, STABLE_WITH_THREADS, STEP_BEFORE_A_SIBLING_THAT_INSTALLS, STEP_INPUT_OFF,
    STEP_INSTALLS, STEP_INSTALLS_BARE, STEP_MISSING_INPUT, TRIPLE_ONLY, TWO_CHANNELS,
    UNKNOWN_CHANNEL,
};

use ci_steps::{
    COVERAGE_DENIES_WARNINGS, Workflow, coverage_problems, linker_install_problems,
    workflow_problems,
};
use config::{CONFIG, Flags, Pin, Problems, THREADS_FLAG, TOOLCHAIN, config_problems};
use make::{
    Assignment, Host, assigned_rustflags, commands_from, development_problems, held_out_problems,
    held_out_target_count, real_make, test_policy_problem,
};

/// Text handed to a case, wrapped so that a case reads as data and the test
/// signatures name what they take rather than passing bare strings around.
#[derive(Clone, Copy)]
struct Fixture(&'static str);

/// The workflow reader a case runs.
#[derive(Clone, Copy)]
enum Reader {
    /// Each `setup-rust` step must pass `install-mold: 'true'` itself.
    InstallInput,
    /// Each coverage step must assign its own `RUSTFLAGS`.
    CoverageAssignment,
}

impl Reader {
    /// Returns the complaints this reader raises about a workflow.
    fn problems(self, workflow: &Workflow) -> Problems {
        match self {
            Self::InstallInput => linker_install_problems(workflow),
            Self::CoverageAssignment => coverage_problems(workflow),
        }
    }
}

/// Turns a list of complaints into a test result.
fn none_of(problems: &Problems) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{problems:#?}"))
    }
}

/// Checks that a fixture configuration draws the expected number of complaints.
fn draws(config: Fixture, pin: Pin, expected: usize) -> Result<(), String> {
    let found = config_problems(config.0, pin)?.len();
    if found == expected {
        Ok(())
    } else {
        Err(format!("{:?}: {found} problems, not {expected}", config.0))
    }
}

/// Scenario: configurations of each shape, read on a nightly and a stable pin.
///
/// Invariant: a compliant nightly file passes, and each way of losing the
/// frontend flag, losing mold, naming mold beyond Linux, or letting a source
/// drift is reported; a stable pin refuses the frontend flag it cannot take.
#[rstest]
#[case::compliant_nightly(Fixture(NIGHTLY_OK), Pin::Nightly, 0)]
#[case::linker_spelled_as_a_pair(Fixture(NIGHTLY_SPELLED_APART), Pin::Nightly, 0)]
#[case::compliant_stable(Fixture(STABLE_OK), Pin::Stable, 0)]
#[case::build_loses_the_frontend(Fixture(BUILD_LOSES_THREADS), Pin::Nightly, 2)]
#[case::linux_loses_the_linker(Fixture(LINUX_LOSES_LINKER), Pin::Nightly, 1)]
#[case::linker_named_in_build(Fixture(LINKER_IN_BUILD), Pin::Nightly, 1)]
#[case::no_build_source(Fixture(NO_BUILD_SOURCE), Pin::Nightly, 1)]
#[case::stable_names_the_frontend(Fixture(STABLE_WITH_THREADS), Pin::Stable, 1)]
#[case::empty_configuration(Fixture(""), Pin::Nightly, 3)]
#[case::linux_selector_covers_one_architecture(Fixture(TRIPLE_ONLY), Pin::Nightly, 1)]
#[case::comments_after_headers_and_entries(Fixture(COMMENTED_OK), Pin::Nightly, 0)]
#[case::a_key_that_only_starts_like_rustflags(Fixture(SIBLING_KEY_OK), Pin::Nightly, 0)]
fn the_configuration_reader_reports_each_defect(
    #[case] config: Fixture,
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

/// Scenario: toolchain files pinning each kind of channel, and files that do
/// not.
///
/// Invariant: only a `nightly` channel reads as nightly, so only it is asked to
/// carry `-Zthreads`; a missing, repeated or unknown channel is an error, not a
/// stable pin by default.
#[rstest]
#[case::nightly(Fixture(NIGHTLY), Some(Pin::Nightly))]
#[case::stable(Fixture(STABLE), Some(Pin::Stable))]
#[case::missing(Fixture(NO_CHANNEL), None)]
#[case::repeated(Fixture(TWO_CHANNELS), None)]
#[case::unknown(Fixture(UNKNOWN_CHANNEL), None)]
fn the_pin_reader_tells_the_channels_apart(
    #[case] toolchain: Fixture,
    #[case] expected: Option<Pin>,
) {
    assert_eq!(Pin::read(toolchain.0).ok(), expected);
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
#[case::plain(Fixture("RUSTFLAGS=\"-D warnings -Zthreads=8\" cargo test"), flags(&["-D", "warnings", THREADS_FLAG], false))]
#[case::inherited_flags_glued_on(
    Fixture("RUSTFLAGS=\"${RUSTFLAGS:+$RUSTFLAGS }-Zthreads=8\" cargo check"),
    flags(&[THREADS_FLAG], true)
)]
#[case::inherited_flags_only(Fixture("RUSTFLAGS=\"${RUSTFLAGS-}\" cargo build --release"), flags(&[], true))]
#[case::inherited_flags_then_a_space(
    Fixture("RUSTFLAGS=\"${RUSTFLAGS-} -Zthreads=8\" cargo check"),
    flags(&[THREADS_FLAG], true)
)]
#[case::no_assignment(Fixture("cargo clippy --all-targets"), Assignment::Unassigned)]
fn the_command_reader_reads_each_assignment(
    #[case] line: Fixture,
    #[case] expected: Assignment,
) -> Result<(), String> {
    if assigned_rustflags(line.0)? == expected {
        Ok(())
    } else {
        Err(format!("`{}` was read wrongly", line.0))
    }
}

/// Scenario: `make -n` output lines whose assignment the reader cannot parse.
///
/// Invariant: each is refused rather than passed, because an assignment in a
/// form the reader does not understand still replaces the configuration.
#[rstest]
#[case::unquoted(Fixture("RUSTFLAGS=-Zthreads=8 cargo test"))]
#[case::unterminated(Fixture("RUSTFLAGS=\"-Zthreads=8 cargo test"))]
#[case::inherited_flags_glued_to_a_flag(Fixture(
    "RUSTFLAGS=\"${RUSTFLAGS-}-Zthreads=8\" cargo test"
))]
fn the_command_reader_refuses_what_it_cannot_parse(#[case] line: Fixture) -> Result<(), String> {
    match assigned_rustflags(line.0) {
        Ok(_) => Err(format!("`{}` was read, not refused", line.0)),
        Err(_) => Ok(()),
    }
}

/// Scenario: workflow steps read by each workflow reader.
///
/// Invariant: a `setup-rust` step must pass `install-mold: 'true'` itself, and a
/// coverage step must assign `RUSTFLAGS` itself and name neither standard flag.
/// Another step's input or assignment does not count, and a comment naming the
/// action is not a step.
#[rstest]
#[case::quoted_true(Reader::InstallInput, Fixture(STEP_INSTALLS), 0)]
#[case::bare_true(Reader::InstallInput, Fixture(STEP_INSTALLS_BARE), 0)]
#[case::missing_input(Reader::InstallInput, Fixture(STEP_MISSING_INPUT), 1)]
#[case::input_off(Reader::InstallInput, Fixture(STEP_INPUT_OFF), 1)]
#[case::input_on_a_sibling_step(
    Reader::InstallInput,
    Fixture(STEP_BEFORE_A_SIBLING_THAT_INSTALLS),
    1
)]
#[case::comment_only(Reader::InstallInput, Fixture(COMMENT_NAMING_THE_ACTION), 0)]
#[case::unassigned(Reader::CoverageAssignment, Fixture(COVERAGE_UNASSIGNED), 1)]
#[case::with_the_frontend_flag(Reader::CoverageAssignment, Fixture(COVERAGE_WITH_THREADS), 1)]
#[case::with_the_linker(Reader::CoverageAssignment, Fixture(COVERAGE_WITH_LINKER), 1)]
#[case::assignment_on_a_sibling_step(
    Reader::CoverageAssignment,
    Fixture(COVERAGE_BORROWING_A_SIBLING),
    1
)]
fn the_workflow_readers_judge_each_step(
    #[case] reader: Reader,
    #[case] workflow: Fixture,
    #[case] expected: usize,
) -> Result<(), String> {
    let found = reader
        .problems(&Workflow {
            file: "fixture.yml",
            text: workflow.0,
        })
        .len();
    if found == expected {
        Ok(())
    } else {
        Err(format!(
            "{:?}: {found} problems, not {expected}",
            workflow.0
        ))
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
    let (problems, read) = development_problems(real_make, Host::Linux, Pin::read(TOOLCHAIN)?)?;
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
    none_of(&development_problems(real_make, Host::Darwin, Pin::read(TOOLCHAIN)?)?.0)
}

/// Coverage measures and release ships, so both stay on the default flags. A
/// repository that lists no such target has nothing local to hold out, and the
/// check then reads no commands; otherwise it must read at least one.
#[test]
fn coverage_and_release_take_neither_flag() -> Result<(), String> {
    let (problems, read) = held_out_problems(real_make)?;
    none_of(&problems)?;
    if held_out_target_count() > 0 && read == 0 {
        return Err(
            "the held-out targets run no cargo command, so the check proves nothing".to_owned(),
        );
    }
    Ok(())
}

/// Scenario: the `test` target's assigned commands, with and without the warning
/// policy, beside a command that never carried it.
///
/// Invariant: `make test` must keep `-D warnings` in at least one of its assigned
/// commands (dropping `$(RUST_FLAGS)` from the command that runs the tests would
/// lose it everywhere), while a version probe or prerequisite build that never
/// carried the policy is not held to it, and another target is not held to it.
#[rstest]
#[case::test_keeps_the_policy(Fixture("test"), &[&["-D", "warnings", THREADS_FLAG][..]], 0)]
#[case::test_spelled_joined(Fixture("test"), &[&["-Dwarnings", THREADS_FLAG][..]], 0)]
#[case::test_drops_the_policy(Fixture("test"), &[&[THREADS_FLAG][..]], 1)]
#[case::test_denies_nothing_useful(Fixture("test"), &[&["-A", "warnings", THREADS_FLAG][..]], 1)]
#[case::a_probe_may_omit_it_beside_the_run(Fixture("test"), &[&[THREADS_FLAG][..], &["-D", "warnings", THREADS_FLAG][..]], 0)]
#[case::build_may_omit_the_policy(Fixture("build"), &[&[THREADS_FLAG][..]], 0)]
#[case::test_assigns_no_command(Fixture("test"), &[], 1)]
fn the_test_target_keeps_the_warning_policy(
    #[case] target: Fixture,
    #[case] commands: &[&[&str]],
    #[case] expected: usize,
) {
    let assigned: Vec<Assignment> = commands
        .iter()
        .map(|words| Assignment::Flags(Flags::from_words(words.iter().copied()), true))
        .collect();
    let found = test_policy_problem(target.0, Host::Darwin, &assigned)
        .into_iter()
        .count();
    assert_eq!(found, expected, "target {}: {commands:?}", target.0);
}

/// Scenario: coverage steps that deny warnings, assign an empty policy, or assign a
/// different one.
///
/// Invariant: the warning policy is pinned both ways, so every case fires in one mode.
/// Where the repository's coverage denies warnings, a step that denies them is accepted
/// and an empty or different policy is refused; where it deliberately does not, a step
/// that starts denying warnings is the drift and is refused, while the others are accepted.
#[rstest]
#[case::denying_warnings(Fixture(COVERAGE_OK), true)]
#[case::an_empty_warning_policy(Fixture(COVERAGE_EMPTY_POLICY), false)]
#[case::a_different_warning_policy(Fixture(COVERAGE_OTHER_POLICY), false)]
#[case::a_lookalike_flag(Fixture(COVERAGE_LOOKALIKE_POLICY), false)]
#[case::a_policy_only_in_a_comment(Fixture(COVERAGE_COMMENTED_POLICY), false)]
#[case::a_comment_after_the_policy(Fixture(COVERAGE_DENYING_WITH_COMMENT), true)]
fn a_coverage_step_keeps_the_repository_warning_policy(
    #[case] workflow: Fixture,
    #[case] denies: bool,
) {
    let found = coverage_problems(&Workflow {
        file: "fixture.yml",
        text: workflow.0,
    })
    .len();
    assert_eq!(
        found,
        usize::from(denies != COVERAGE_DENIES_WARNINGS),
        "workflow:\n{}",
        workflow.0
    );
}
