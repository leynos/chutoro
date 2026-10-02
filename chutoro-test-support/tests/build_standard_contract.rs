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

#[path = "build_standard_support/ci_steps.rs"]
mod ci_steps;
#[path = "build_standard_support/config.rs"]
mod config;
#[path = "build_standard_support/exhaustive.rs"]
mod exhaustive;
#[path = "build_standard_support/fixtures.rs"]
mod fixtures;
#[path = "build_standard_support/make.rs"]
mod make;
use std::fmt::Write as _;

use rstest::rstest;

use fixtures::{
    BUILD_LOSES_THREADS, COMMENT_NAMING_THE_ACTION, COMMENTED_OK, COVERAGE_BORROWING_A_SIBLING,
    COVERAGE_EMPTY_POLICY, COVERAGE_OK, COVERAGE_OTHER_POLICY, COVERAGE_UNASSIGNED,
    COVERAGE_WITH_LINKER, COVERAGE_WITH_THREADS, LINKER_IN_BUILD, LINUX_LOSES_LINKER, NIGHTLY,
    NIGHTLY_OK, NIGHTLY_SPELLED_APART, NO_BUILD_SOURCE, NO_CHANNEL, SIBLING_KEY_OK, SPREAD_ARRAY,
    STABLE, STABLE_OK, STABLE_WITH_THREADS, STEP_BEFORE_A_SIBLING_THAT_INSTALLS, STEP_INPUT_OFF,
    STEP_INSTALLS, STEP_INSTALLS_BARE, STEP_MISSING_INPUT, TRIPLE_ONLY, TWO_CHANNELS,
    UNKNOWN_CHANNEL,
};

use ci_steps::{Workflow, coverage_problems, linker_install_problems, workflow_problems};
use config::{CONFIG, Flags, Pin, Problems, THREADS_FLAG, TOOLCHAIN, config_problems};
use make::{
    Assignment, Host, assigned_rustflags, commands_from, development_problem, development_problems,
    held_out_problems, held_out_target_count, real_make,
};

/// Turns a list of complaints into a test result.
fn none_of(problems: &Problems) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{problems:#?}"))
    }
}

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
#[case::linux_selector_covers_one_architecture(TRIPLE_ONLY, Pin::Nightly, 1)]
#[case::comments_after_headers_and_entries(COMMENTED_OK, Pin::Nightly, 0)]
#[case::a_key_that_only_starts_like_rustflags(SIBLING_KEY_OK, Pin::Nightly, 0)]
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
#[case::inherited_flags_then_a_space(
    "RUSTFLAGS=\"${RUSTFLAGS-} -Zthreads=8\" cargo check",
    flags(&[THREADS_FLAG], true)
)]
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
#[case::inherited_flags_glued_to_a_flag("RUSTFLAGS=\"${RUSTFLAGS-}-Zthreads=8\" cargo test")]
fn the_command_reader_refuses_what_it_cannot_parse(#[case] line: &str) -> Result<(), String> {
    match assigned_rustflags(line) {
        Ok(_) => Err(format!("`{line}` was read, not refused")),
        Err(_) => Ok(()),
    }
}

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
    let found = linker_install_problems(&Workflow {
        file: "fixture.yml",
        text: workflow,
    })
    .len();
    if found == expected {
        Ok(())
    } else {
        Err(format!("{workflow:?}: {found} problems, not {expected}"))
    }
}

/// Scenario: coverage steps with and without an explicit assignment.
///
/// Invariant: the step assigns `RUSTFLAGS` itself and names neither standard
/// flag; a sibling step's assignment does not count.
#[rstest]
#[case::assigned(COVERAGE_OK, 0)]
#[case::an_empty_warning_policy(COVERAGE_EMPTY_POLICY, 1)]
#[case::a_different_warning_policy(COVERAGE_OTHER_POLICY, 1)]
#[case::unassigned(COVERAGE_UNASSIGNED, 1)]
#[case::with_the_frontend_flag(COVERAGE_WITH_THREADS, 1)]
#[case::with_the_linker(COVERAGE_WITH_LINKER, 1)]
#[case::assignment_on_a_sibling_step(COVERAGE_BORROWING_A_SIBLING, 1)]
fn the_coverage_reader_wants_an_explicit_assignment(
    #[case] workflow: &str,
    #[case] expected: usize,
) -> Result<(), String> {
    let found = coverage_problems(&Workflow {
        file: "fixture.yml",
        text: workflow,
    })
    .len();
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

/// Scenario: a development command that assigns the standard flags, with and
/// without the warning policy, for the `test` target and another target.
///
/// Invariant: the `test` target must keep `-D warnings` beside the standard flags
/// (dropping `$(RUST_FLAGS)` would silently stop denying warnings), while a target
/// that never carried the policy is not held to it.
#[rstest]
#[case::test_keeps_the_policy("test", &["-D", "warnings", THREADS_FLAG], 0)]
#[case::test_spelled_joined("test", &["-Dwarnings", THREADS_FLAG], 0)]
#[case::test_drops_the_policy("test", &[THREADS_FLAG], 1)]
#[case::test_denies_nothing_useful("test", &["-A", "warnings", THREADS_FLAG], 1)]
#[case::build_may_omit_the_policy("build", &[THREADS_FLAG], 0)]
fn the_test_target_keeps_the_warning_policy(
    #[case] target: &str,
    #[case] words: &[&str],
    #[case] expected: usize,
) {
    let assignment = Assignment::Flags(Flags::from_words(words.iter().copied()), true);
    let found = development_problem(target, Host::Darwin, Pin::Nightly, &assignment)
        .into_iter()
        .count();
    assert_eq!(found, expected, "target {target}: {words:?}");
}

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

/// A fake runner for a target that is not defined.
fn undefined_make(target: &str, _host: Host) -> Result<String, String> {
    Err(format!("`make -n {target}` failed, so it is not defined"))
}

/// Scenario: the policy checks fed canned `make -n` text through the injected
/// runner, so no process runs.
///
/// Invariant: a compliant command raises no complaint on either host, a command
/// that drops the caller's flags raises one per target, and a runner error
/// reaches the caller instead of being read as an empty output.
#[test]
fn the_policy_checks_run_against_an_injected_runner() {
    let pin = Pin::Nightly;
    for host in [Host::Linux, Host::Darwin] {
        let (problems, read) =
            development_problems(compliant_make, host, pin).expect("a compliant fake reads");
        assert!(problems.is_empty(), "a compliant fake raised {problems:?}");
        assert!(read > 0, "the fake's commands were not read");
    }
    let (dropped, _) =
        development_problems(dropping_make, Host::Linux, pin).expect("the dropping fake reads");
    assert!(
        !dropped.is_empty(),
        "a command that drops the caller's flags passed"
    );
    assert!(
        development_problems(undefined_make, Host::Linux, pin).is_err(),
        "a runner error was swallowed"
    );
    assert!(
        held_out_problems(undefined_make).is_err(),
        "a held-out runner error was swallowed"
    );
}
