//! Bounded exhaustive test for the workflow-step readers: every small workflow drawn from a
//! few step shapes is rendered, and each reader's complaints are checked against what the
//! generator knows, without a second reader to trust.

use super::{
    ci_steps::{COVERAGE_DENIES_WARNINGS, Workflow, coverage_problems, linker_install_problems},
    exhaustive::sequences,
};

/// One step a generated workflow can hold, and what the reader must make of it.
#[derive(Clone, Copy, Debug)]
enum StepKind {
    /// A `setup-rust` step that passes `install-mold: 'true'`.
    SetupInstalling,
    /// A `setup-rust` step that passes `install-mold: true`, unquoted.
    SetupInstallingBare,
    /// A `setup-rust` step with no `install-mold`, only a comment that mentions it.
    SetupNotInstalling,
    /// A `generate-coverage` step that assigns its own `RUSTFLAGS`.
    CoverageAssigning,
    /// A `generate-coverage` step with no `RUSTFLAGS`, only a comment that mentions it.
    CoverageNotAssigning,
    /// Another action's step that passes `install-mold` and `RUSTFLAGS` itself.
    SiblingWithBoth,
    /// A comment that names both actions, between steps.
    CommentNamingBoth,
}

const SETUP: &str =
    "- uses: o/shared-actions/.github/actions/setup-rust@0123456789012345678901234567890123456789";
const COVER: &str = "- uses: o/shared-actions/.github/actions/generate-coverage@0123456789012345678901234567890123456789";

/// Which complaint a generated step must draw, and the offset of its `uses:` line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Draws {
    Nothing,
    Linker(usize),
    Coverage(usize),
}

impl StepKind {
    const ALL: [Self; 7] = [
        Self::SetupInstalling,
        Self::SetupInstallingBare,
        Self::SetupNotInstalling,
        Self::CoverageAssigning,
        Self::CoverageNotAssigning,
        Self::SiblingWithBoth,
        Self::CommentNamingBoth,
    ];

    /// Returns the step's lines, with `@FLAGS@` standing for a coverage policy the repository accepts.
    const fn template(self) -> &'static [&'static str] {
        match self {
            Self::SetupInstalling => &[SETUP, "  with:", "    install-mold: 'true'"],
            Self::SetupInstallingBare => &[SETUP, "  with:", "    install-mold: true"],
            Self::SetupNotInstalling => &[
                "- name: Set up Rust",
                "  uses: o/shared-actions/.github/actions/setup-rust@0123456789012345678901234567890123456789",
                "  with:",
                "    # install-mold: 'true'",
            ],
            Self::CoverageAssigning => &[COVER, "  env:", "    RUSTFLAGS: @FLAGS@"],
            Self::CoverageNotAssigning => &[COVER, "  env:", "    # RUSTFLAGS: -D warnings"],
            Self::SiblingWithBoth => &[
                "- uses: actions/checkout@v4",
                "  with:",
                "    install-mold: 'true'",
                "  env:",
                "    RUSTFLAGS: -D warnings",
            ],
            Self::CommentNamingBoth => &[
                "# uses: setup-rust@ and generate-coverage@ with install-mold: 'true' and RUSTFLAGS: x",
            ],
        }
    }

    /// Returns the complaint the step must draw, and where its `uses:` line sits.
    const fn draws(self) -> Draws {
        match self {
            Self::SetupNotInstalling => Draws::Linker(1),
            Self::CoverageNotAssigning => Draws::Coverage(0),
            _ => Draws::Nothing,
        }
    }

    /// Returns the step's lines, indented by `pad`.
    fn lines(self, pad: &str) -> Vec<String> {
        let flags = if COVERAGE_DENIES_WARNINGS {
            "-D warnings"
        } else {
            "-C opt-level=1"
        };
        self.template()
            .iter()
            .map(|line| format!("{pad}{}", line.replace("@FLAGS@", flags)))
            .collect()
    }
}

/// A generated workflow, and the lines at which its steps must draw a complaint.
struct Generated {
    text: String,
    unlinked: Vec<usize>,
    unassigned: Vec<usize>,
}

/// Renders a workflow from a sequence of steps at one indent.
fn generate(steps: &[StepKind], pad: &str) -> Generated {
    let mut generated = Generated {
        text: String::from("jobs:\n  build:\n    steps:\n"),
        unlinked: Vec::new(),
        unassigned: Vec::new(),
    };
    let mut line_no = 3;
    for kind in steps {
        match kind.draws() {
            Draws::Linker(offset) => generated.unlinked.push(line_no + offset + 1),
            Draws::Coverage(offset) => generated.unassigned.push(line_no + offset + 1),
            Draws::Nothing => {}
        }
        let lines = kind.lines(pad);
        line_no += lines.len();
        generated
            .text
            .extend(lines.iter().map(|line| format!("{line}\n")));
    }
    generated
}

/// Scenario: every workflow of up to three steps drawn from installing and bare setup steps,
/// setup and coverage steps that only mention their input in a comment, a sibling step that
/// passes both inputs itself, and a comment naming both actions, at two step indents.
///
/// Invariant: the reader complains about exactly the steps that lack their own input, each
/// at its own `uses:` line, whatever sits before or after: a sibling step's input never
/// rescues the step beside it, and a comment never makes a step or hides one.
#[test]
fn the_workflow_readers_judge_each_step_by_its_own_lines_over_every_small_workflow() {
    for pad in ["      ", "    "] {
        for steps in sequences(&StepKind::ALL, 3) {
            let generated = generate(&steps, pad);
            let workflow = Workflow {
                file: "generated.yml",
                text: &generated.text,
            };
            let linker = line_numbers(&linker_install_problems(&workflow));
            let coverage = line_numbers(&coverage_problems(&workflow));
            assert_eq!(
                linker, generated.unlinked,
                "setup-rust steps, steps {steps:?}:\n{}",
                generated.text
            );
            assert_eq!(
                coverage, generated.unassigned,
                "coverage steps, steps {steps:?}:\n{}",
                generated.text
            );
        }
    }
}

/// Returns the line each `file:line: ...` complaint names.
fn line_numbers(problems: &[String]) -> Vec<usize> {
    problems
        .iter()
        .filter_map(|problem| problem.split(':').nth(1).and_then(|line| line.parse().ok()))
        .collect()
}
