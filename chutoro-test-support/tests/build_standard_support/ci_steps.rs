//! Reader for the CI half of the build standard: every workflow that builds
//! under the standard installs mold through `setup-rust`'s `install-mold`
//! input, so the Linux jobs have the linker the configuration names.
//!
//! The workflows are read as text, one step at a time. Release workflows are
//! not listed: a release stays on the platform linker and never uses mold.

use super::config::Problems;

/// The workflows that set up Rust and build under the standard, as name and
/// text. The list is this repository's own, so a workflow that stops setting up
/// Rust fails the contract rather than dropping out of it.
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
///
/// The step starts at the nearest list item to the left of the line, which also
/// holds for a value folded onto the line after `uses: >-`.
fn step_lines<'a>(lines: &[&'a str], at: usize) -> Vec<&'a str> {
    let here = lines.get(at).copied().unwrap_or_default();
    let start = (0..=at)
        .rev()
        .find(|&index| {
            lines.get(index).is_some_and(|line| {
                opens_step(line) && (index == at || indent(line) < indent(here))
            })
        })
        .unwrap_or(at);
    let step_indent = lines.get(start).map_or(0, |line| indent(line));
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
fn installs_linker(step: &[&str]) -> bool {
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
pub fn linker_install_problems(name: &str, workflow: &str) -> Problems {
    let lines: Vec<&str> = workflow.lines().collect();
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains("setup-rust@") && !line.trim_start().starts_with('#'))
        .filter(|(at, _)| !installs_linker(&step_lines(&lines, *at)))
        .map(|(at, _)| {
            format!(
                "{name}:{}: a setup-rust step does not pass `install-mold: 'true'`",
                at + 1
            )
        })
        .collect()
}

/// Returns the `RUSTFLAGS` value a step assigns, if it has an `RUSTFLAGS:`
/// line.
fn step_rustflags(step: &[&str]) -> Option<String> {
    step.iter()
        .map(|line| line.trim())
        .find_map(|line| line.strip_prefix("RUSTFLAGS:"))
        .map(|value| value.trim().to_owned())
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
pub fn coverage_problems(name: &str, workflow: &str) -> Problems {
    let lines: Vec<&str> = workflow.lines().collect();
    let mut problems = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        if !line.contains("generate-coverage@") || line.trim_start().starts_with('#') {
            continue;
        }
        let step = step_lines(&lines, at);
        match step_rustflags(&step) {
            None => problems.push(format!(
                "{name}:{}: a coverage step does not assign RUSTFLAGS",
                at + 1
            )),
            Some(value) if value.contains("-Zthreads") || value.contains("mold") => {
                problems.push(format!(
                    "{name}:{}: a coverage step assigns a standard flag: {value}",
                    at + 1
                ));
            }
            Some(_) => {}
        }
    }
    problems
}

/// Returns the complaints about one listed workflow: a step without the input,
/// or no `setup-rust` step at all, which would leave the check reading nothing.
fn listed_problems(name: &str, workflow: &str) -> Problems {
    let mut problems = linker_install_problems(name, workflow);
    problems.extend(coverage_problems(name, workflow));
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
