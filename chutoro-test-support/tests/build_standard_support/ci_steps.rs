//! Reader for the CI half of the build standard: every workflow that builds under
//! the standard installs mold through `setup-rust`'s `install-mold` input, so the
//! Linux jobs have the linker the configuration names, and every coverage step
//! assigns `RUSTFLAGS` itself, without a standard flag.
//!
//! The workflows are read as text, one step at a time. Release workflows are not
//! listed: a release stays on the platform linker and never uses mold.

use super::config::{Flags, Problems, THREADS_FLAG};

/// The workflows that set up Rust and build under the standard, as name and text.
/// The list is this repository's own, so a workflow that stops setting up Rust
/// fails the contract rather than dropping out of it.
pub const WORKFLOWS: &[(&str, &str)] = &[
    (
        "benchmark-regressions.yml",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/benchmark-regressions.yml"
        )),
    ),
    (
        "ci.yml",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/ci.yml"
        )),
    ),
    (
        "coverage-main.yml",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/coverage-main.yml"
        )),
    ),
    (
        "kani-pr.yml",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/kani-pr.yml"
        )),
    ),
    (
        "nightly-kani.yml",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/nightly-kani.yml"
        )),
    ),
    (
        "property-tests.yml",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/property-tests.yml"
        )),
    ),
];

/// A workflow file: its name for complaints, and its text.
#[derive(Clone, Copy)]
pub struct Workflow<'a> {
    pub file: &'a str,
    pub text: &'a str,
}

/// The actions whose steps the standard constrains.
#[derive(Clone, Copy)]
enum Action {
    SetupRust,
    GenerateCoverage,
}

impl Action {
    /// Returns the text that marks a `uses:` line as this action.
    const fn marker(self) -> &'static str {
        match self {
            Self::SetupRust => "setup-rust@",
            Self::GenerateCoverage => "generate-coverage@",
        }
    }
}

/// Whether this repository's coverage steps assign a `RUSTFLAGS` that denies warnings. The
/// policy is pinned both ways: a step that stops denying warnings is a regression where this is
/// `true`, and a step that starts denying them is drift where a repository deliberately assigns
/// other flags (an alternative linker, a different frontend flag) and this is `false`.
pub const COVERAGE_DENIES_WARNINGS: bool = true;

/// A step of a workflow file, found by the action it uses.
struct Step<'a> {
    file: &'a str,
    line: usize,
    lines: Vec<&'a str>,
}

impl Step<'_> {
    /// Returns where the step is, for a complaint.
    fn location(&self) -> String {
        format!("{}:{}", self.file, self.line)
    }

    /// Returns whether the step passes `install-mold: 'true'` (quoted or bare).
    fn passes_the_install_input(&self) -> bool {
        self.lines
            .iter()
            .any(|line| squeezed(line) == "install-mold:true")
    }

    /// Returns the value of the step's own `RUSTFLAGS:` line, if it has one.
    fn rustflags(&self) -> Option<&str> {
        self.lines
            .iter()
            .find_map(|line| line.trim().strip_prefix("RUSTFLAGS:"))
            .map(str::trim)
    }

    /// Returns the complaint about a `setup-rust` step that installs no linker.
    fn install_problem(&self) -> Option<String> {
        (!self.passes_the_install_input()).then(|| {
            format!(
                "{}: a setup-rust step does not pass `install-mold: 'true'`",
                self.location()
            )
        })
    }

    /// Returns the complaint about a coverage step that leaves `RUSTFLAGS` to
    /// the setup action, or assigns one that names a standard flag.
    fn coverage_problem(&self) -> Option<String> {
        let Some(value) = self.rustflags() else {
            return Some(format!(
                "{}: a coverage step does not assign RUSTFLAGS",
                self.location()
            ));
        };
        // An inline YAML comment is not part of the value, and the warning policy is read as
        // whole flags, so `-D warnings-extra` and `# -D warnings` do not deny anything.
        let words: Vec<&str> = value
            .split_whitespace()
            .take_while(|word| !word.starts_with('#'))
            .map(|word| word.trim_matches(['"', '\'']))
            .collect();
        let names_a_standard_flag = words
            .iter()
            .any(|word| word.contains(THREADS_FLAG) || word.contains("mold"));
        let denies_warnings = Flags::from_words(words.iter().copied()).denies_warnings();
        let reason = if names_a_standard_flag {
            "assigns a standard flag"
        } else if denies_warnings != COVERAGE_DENIES_WARNINGS {
            "assigns a warning policy that differs from this repository's"
        } else {
            return None;
        };
        Some(format!(
            "{}: a coverage step {reason}: {value}",
            self.location()
        ))
    }
}

/// Returns a line without spaces and quote marks.
fn squeezed(line: &str) -> String {
    line.chars()
        .filter(|c| !matches!(c, ' ' | '\'' | '"'))
        .collect()
}

/// Returns the number of leading spaces on a line.
fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Returns whether a line opens a step: a `- ` list item.
fn opens_step(line: &str) -> bool {
    line.trim_start().starts_with("- ")
}

/// Returns the lines of the step holding the line at `at`: from the step's own
/// list item to the line before the next step, or to the end of the job.
///
/// The step starts at the nearest list item to the left of the line, which also
/// holds for a value folded onto the line after `uses: >-`.
fn step_lines<'a>(lines: &[&'a str], at: usize) -> Vec<&'a str> {
    let here = lines.get(at).copied().unwrap_or_default();
    let starts_the_step = |index: &usize| {
        lines
            .get(*index)
            .is_some_and(|line| opens_step(line) && (*index == at || indent(line) < indent(here)))
    };
    let start = (0..=at).rev().find(starts_the_step).unwrap_or(at);
    let step_indent = lines.get(start).map_or(0, |line| indent(line));
    let ends_the_step = |index: &usize| {
        lines.get(*index).is_some_and(|line| {
            !line.trim().is_empty()
                && (indent(line) < step_indent || (opens_step(line) && indent(line) <= step_indent))
        })
    };
    let end = (at + 1..lines.len())
        .find(ends_the_step)
        .unwrap_or(lines.len());
    lines.get(start..end).unwrap_or_default().to_vec()
}

/// Returns every step of a workflow that uses an action, skipping comments.
fn steps_using<'a>(workflow: &Workflow<'a>, action: Action) -> Vec<Step<'a>> {
    let lines: Vec<&str> = workflow.text.lines().collect();
    let uses = |line: &&str| line.contains(action.marker()) && !line.trim_start().starts_with('#');
    let found: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| uses(line))
        .map(|(at, _)| at)
        .collect();
    found
        .into_iter()
        .map(|at| Step {
            file: workflow.file,
            line: at + 1,
            lines: step_lines(&lines, at),
        })
        .collect()
}

/// Returns the complaint about each `setup-rust` step in one workflow that does
/// not pass `install-mold: 'true'`.
///
/// ```text
/// - uses: org/shared-actions/.github/actions/setup-rust@<sha>
///   with:
///     install-mold: 'true'      -> no complaint
/// ```
pub fn linker_install_problems(workflow: &Workflow) -> Problems {
    steps_using(workflow, Action::SetupRust)
        .iter()
        .filter_map(Step::install_problem)
        .collect()
}

/// Returns the complaint about each coverage step in one workflow that does not
/// assign `RUSTFLAGS` itself, or assigns one that names a standard flag.
///
/// A coverage build is a measurement, so it takes neither the frontend flag nor
/// mold. The exception is explicit in the step, not a side effect of whatever
/// the setup action exports.
///
/// ```text
/// - uses: org/shared-actions/.github/actions/generate-coverage@<sha>
///   env:
///     RUSTFLAGS: -D warnings      -> no complaint
/// ```
pub fn coverage_problems(workflow: &Workflow) -> Problems {
    steps_using(workflow, Action::GenerateCoverage)
        .iter()
        .filter_map(Step::coverage_problem)
        .collect()
}

/// Returns the complaints about one listed workflow: its steps, or no
/// `setup-rust` step at all, which would leave the check reading nothing.
fn listed_problems(workflow: &Workflow) -> Problems {
    let mut problems = linker_install_problems(workflow);
    problems.extend(coverage_problems(workflow));
    if steps_using(workflow, Action::SetupRust).is_empty() {
        let message = format!(
            "{}: the listed workflow has no setup-rust step, so the check proves nothing",
            workflow.file
        );
        problems.push(message);
    }
    problems
}

/// Returns every complaint about the listed workflows.
pub fn workflow_problems() -> Problems {
    let listed = WORKFLOWS
        .iter()
        .map(|&(file, text)| Workflow { file, text });
    listed
        .flat_map(|workflow| listed_problems(&workflow))
        .collect()
}
