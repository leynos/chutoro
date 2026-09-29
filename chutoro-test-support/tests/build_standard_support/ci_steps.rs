//! Reader for the CI half of the build standard: every workflow that builds under
//! the standard installs mold through `setup-rust`'s `install-mold` input, so the
//! Linux jobs have the linker the configuration names, and every coverage step
//! assigns `RUSTFLAGS` itself, without a standard flag.
//!
//! The workflows are read as text, one step at a time. Release workflows are not
//! listed: a release stays on the platform linker and never uses mold.

use super::config::{Problems, THREADS_FLAG};

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
        let names_a_standard_flag = value.contains(THREADS_FLAG) || value.contains("mold");
        names_a_standard_flag.then(|| {
            format!(
                "{}: a coverage step assigns a standard flag: {value}",
                self.location()
            )
        })
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
fn steps_using<'a>(file: &'a str, workflow: &'a str, action: &str) -> Vec<Step<'a>> {
    let lines: Vec<&str> = workflow.lines().collect();
    let uses = |line: &&str| line.contains(action) && !line.trim_start().starts_with('#');
    let found: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| uses(line))
        .map(|(at, _)| at)
        .collect();
    found
        .into_iter()
        .map(|at| Step {
            file,
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
pub fn linker_install_problems(file: &str, workflow: &str) -> Problems {
    steps_using(file, workflow, "setup-rust@")
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
pub fn coverage_problems(file: &str, workflow: &str) -> Problems {
    steps_using(file, workflow, "generate-coverage@")
        .iter()
        .filter_map(Step::coverage_problem)
        .collect()
}

/// Returns the complaints about one listed workflow: its steps, or no
/// `setup-rust` step at all, which would leave the check reading nothing.
fn listed_problems(file: &str, workflow: &str) -> Problems {
    let mut problems = linker_install_problems(file, workflow);
    problems.extend(coverage_problems(file, workflow));
    if steps_using(file, workflow, "setup-rust@").is_empty() {
        problems.push(format!(
            "{file}: the listed workflow has no setup-rust step, so the check proves nothing"
        ));
    }
    problems
}

/// Returns every complaint about the listed workflows.
pub fn workflow_problems() -> Problems {
    WORKFLOWS
        .iter()
        .flat_map(|(file, workflow)| listed_problems(file, workflow))
        .collect()
}
