//! Reader for the CI half of the build standard: every workflow that builds under
//! the standard installs mold through `setup-rust`'s `install-mold` input, so the
//! Linux jobs have the linker the configuration names.
//!
//! The workflows are read as text, one step at a time. Release workflows are not
//! listed: a release stays on the platform linker and never uses mold.

use super::config::Problems;

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
fn step_lines<'a>(lines: &[&'a str], at: usize) -> Vec<&'a str> {
    let here = lines.get(at).copied().unwrap_or_default();
    let step_indent = if opens_step(here) {
        indent(here)
    } else {
        indent(here).saturating_sub(2)
    };
    let start = (0..=at)
        .rev()
        .find(|&index| {
            lines
                .get(index)
                .is_some_and(|line| opens_step(line) && indent(line) == step_indent)
        })
        .unwrap_or(at);
    let end = (at + 1..lines.len())
        .find(|&index| {
            lines.get(index).is_some_and(|line| {
                !line.trim().is_empty()
                    && (indent(line) < step_indent
                        || (opens_step(line) && indent(line) <= step_indent))
            })
        })
        .unwrap_or(lines.len());
    lines.get(start..end).unwrap_or_default().to_vec()
}

/// Returns whether a step passes `install-mold: 'true'` (quoted or bare).
fn installs_mold(step: &[&str]) -> bool {
    step.iter().any(|line| {
        let squeezed: String = line
            .chars()
            .filter(|c| !matches!(c, ' ' | '\'' | '"'))
            .collect();
        squeezed == "install-mold:true"
    })
}

/// Returns the complaint about each `setup-rust` step in one workflow that does
/// not pass `install-mold: 'true'`.
///
/// ```text
/// - uses: org/shared-actions/.github/actions/setup-rust@<sha>
///   with:
///     install-mold: 'true'      -> no complaint
/// ```
pub fn install_mold_problems(name: &str, workflow: &str) -> Problems {
    let lines: Vec<&str> = workflow.lines().collect();
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains("setup-rust@") && !line.trim_start().starts_with('#'))
        .filter(|(at, _)| !installs_mold(&step_lines(&lines, *at)))
        .map(|(at, _)| {
            format!(
                "{name}:{}: a setup-rust step does not pass `install-mold: 'true'`",
                at + 1
            )
        })
        .collect()
}

/// Returns the complaints about one listed workflow: a step without the input,
/// or no `setup-rust` step at all, which would leave the check reading nothing.
fn listed_problems(name: &str, workflow: &str) -> Problems {
    let mut problems = install_mold_problems(name, workflow);
    if !workflow.contains("setup-rust@") {
        problems.push(format!(
            "{name}: the listed workflow has no setup-rust step, so the check proves nothing"
        ));
    }
    problems
}

/// Returns every complaint about the listed workflows.
pub fn workflow_problems() -> Problems {
    WORKFLOWS
        .iter()
        .flat_map(|(name, text)| listed_problems(name, text))
        .collect()
}
