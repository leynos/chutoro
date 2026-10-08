//! Contract tests for the workflow steps that run `cargo` directly.
//!
//! The Makefile recipes and the `setup-rust` steps are held by the build
//! standard's own contract. A step that calls `cargo` itself, in the nightly
//! portable SIMD lane and the property suite, is held here. An assigned
//! `RUSTFLAGS` replaces every `rustflags` table in `.cargo/config.toml`, and
//! `setup-rust` exports one of its own, so such a step carries the linker flag
//! beside the deny that the action would otherwise have supplied.
//!
//! The reader is text-based, like the rest of the contract. It judges each
//! step by its own lines, so a step cannot borrow a sibling's `env:`, and it
//! skips comments, so prose naming `cargo` neither adds a step nor satisfies
//! one. Fixtures and a bounded exhaustive sweep come first, so the rule cannot
//! pass by finding no steps.

/// The linker flag the standard repeats wherever `RUSTFLAGS` is assigned.
const LINKER_FLAG: &str = "-Clink-arg=-fuse-ld=mold";

/// A workflow this contract reads, with the number of direct `cargo` steps it
/// must hold. The count is pinned so that a renamed or removed step fails the
/// contract instead of leaving it with nothing to judge.
struct Listed {
    file: &'static str,
    text: &'static str,
    cargo_steps: usize,
}

const CI: Listed = Listed {
    file: "ci.yml",
    text: include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../.github/workflows/ci.yml"
    )),
    cargo_steps: 1,
};

const COVERAGE_MAIN: Listed = Listed {
    file: "coverage-main.yml",
    text: include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../.github/workflows/coverage-main.yml"
    )),
    cargo_steps: 1,
};

const LISTED: &[Listed] = &[
    Listed {
        file: "nightly-portable-simd.yml",
        text: include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/nightly-portable-simd.yml"
        )),
        cargo_steps: 2,
    },
    Listed {
        file: "property-tests.yml",
        text: include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.github/workflows/property-tests.yml"
        )),
        cargo_steps: 1,
    },
    CI,
    COVERAGE_MAIN,
];

/// A step of a workflow: where it starts, and its lines.
struct Step<'a> {
    line: usize,
    lines: Vec<&'a str>,
}

/// Returns the number of leading spaces on a line.
fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Returns whether a line carries nothing a step is judged by.
fn is_blank_or_comment(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty() || trimmed.starts_with('#')
}

/// Returns the steps of every `steps:` list in a workflow.
///
/// A list ends at the first line indented less than its items. A step runs
/// from its `- ` item to the line before the next item at the same
/// indentation, or to the end of the list.
fn steps_of(text: &str) -> Vec<Step<'_>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut steps = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        if line.trim() == "steps:" {
            steps.extend(list_after(&lines, at));
        }
    }
    steps
}

/// Returns the steps of the list that follows the `steps:` key at `key`.
fn list_after<'a>(lines: &[&'a str], key: usize) -> Vec<Step<'a>> {
    let significant = |index: &usize| lines.get(*index).is_some_and(|l| !is_blank_or_comment(l));
    let Some(first) = (key + 1..lines.len()).find(significant) else {
        return Vec::new();
    };
    let item_indent = lines.get(first).map_or(0, |item| indent(item));
    let opens_item = |index: &usize| {
        lines.get(*index).is_some_and(|l| {
            !is_blank_or_comment(l) && indent(l) == item_indent && l.trim_start().starts_with("- ")
        })
    };
    let end = (first + 1..lines.len())
        .find(|index| {
            significant(index) && lines.get(*index).is_some_and(|l| indent(l) < item_indent)
        })
        .unwrap_or(lines.len());
    let starts: Vec<usize> = (first..end).filter(opens_item).collect();
    starts
        .iter()
        .zip(starts.iter().skip(1).chain(std::iter::once(&end)))
        .map(|(start, next)| Step {
            line: start + 1,
            lines: lines.get(*start..*next).unwrap_or_default().to_vec(),
        })
        .collect()
}

/// Returns whether a line invokes `cargo` as a command: a whole word, outside
/// a comment and outside a step's `name:`.
fn invokes_cargo(line: &str) -> bool {
    let trimmed = line.trim_start().trim_start_matches("- ").trim_start();
    !trimmed.starts_with('#')
        && !trimmed.starts_with("name:")
        && trimmed.split_whitespace().any(|word| word == "cargo")
}

impl<'a> Step<'a> {
    /// Returns whether the step runs `cargo` itself.
    fn runs_cargo(&self) -> bool {
        self.lines.iter().any(|line| invokes_cargo(line))
    }

    /// Returns the value of the step's own `RUSTFLAGS:` line, without any
    /// inline YAML comment.
    fn rustflags(&self) -> Option<&'a str> {
        self.lines
            .iter()
            .find_map(|line| line.trim().strip_prefix("RUSTFLAGS:"))
            .map(|value| value.split(" #").next().unwrap_or_default().trim())
    }

    /// Returns the complaint about a step whose `RUSTFLAGS` lacks the linker
    /// flag or the warning deny, or that assigns none.
    fn problem(&self, file: &str) -> Option<String> {
        let Some(value) = self.rustflags() else {
            return Some(format!(
                "{file}:{}: a step runs cargo without assigning RUSTFLAGS, so it takes the \
                 setup action's flags and loses the linker",
                self.line
            ));
        };
        let words: Vec<&str> = value.split_whitespace().collect();
        let denies = words
            .windows(2)
            .any(|pair| matches!(pair, ["-D", "warnings"]))
            || words.contains(&"-Dwarnings");
        let links = words.contains(&LINKER_FLAG);
        (!(denies && links)).then(|| {
            format!(
                "{file}:{}: RUSTFLAGS `{value}` must deny warnings and carry `{LINKER_FLAG}`",
                self.line
            )
        })
    }
}

/// The step that `ci.yml` runs and `coverage-main.yml` warms the compiler cache for.
const WARMED_STEP: &str = "Dense stable SIMD gating";

/// Returns the `RUSTFLAGS` a workflow's step of the given name assigns, if the
/// step exists and assigns one.
fn rustflags_of_step<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    steps_of(text)
        .into_iter()
        .find(|step| {
            step.lines.iter().any(|line| {
                line.trim_start()
                    .trim_start_matches("- ")
                    .strip_prefix("name:")
                    .is_some_and(|value| value.trim() == name)
            })
        })
        .and_then(|step| step.rustflags())
}

/// Returns the number of steps in a workflow that run `cargo` directly.
fn cargo_step_count(text: &str) -> usize {
    steps_of(text)
        .iter()
        .filter(|step| step.runs_cargo())
        .count()
}

/// Returns the complaint about each direct `cargo` step in a workflow.
fn problems_in(file: &str, text: &str) -> Vec<String> {
    steps_of(text)
        .iter()
        .filter(|step| step.runs_cargo())
        .filter_map(|step| step.problem(file))
        .collect()
}

/// Returns the complaints about a listed workflow: each step's, and a count
/// that differs from the pinned one.
fn listed_problems(listed: &Listed) -> Vec<String> {
    let mut problems = problems_in(listed.file, listed.text);
    let found = cargo_step_count(listed.text);
    if found != listed.cargo_steps {
        problems.push(format!(
            "{}: expected {} direct cargo step(s), found {found}; a renamed or removed step \
             leaves this contract judging nothing",
            listed.file, listed.cargo_steps
        ));
    }
    problems
}

const COMPLIANT: &str = "\
jobs:
  build:
    steps:
      - name: Test
        env:
          RUSTFLAGS: -D warnings -Clink-arg=-fuse-ld=mold
        run: cargo test
";

const NO_ENV: &str = "\
jobs:
  build:
    steps:
      - name: Test
        run: cargo test
";

const LACKS_LINKER: &str = "\
jobs:
  build:
    steps:
      - name: Test
        env:
          RUSTFLAGS: -D warnings
        run: cargo test
";

const LACKS_DENY: &str = "\
jobs:
  build:
    steps:
      - name: Test
        env:
          RUSTFLAGS: -Clink-arg=-fuse-ld=mold
        run: cargo test
";

const LOOKALIKE_DENY: &str = "\
jobs:
  build:
    steps:
      - name: Test
        env:
          RUSTFLAGS: -D warnings-extra -Clink-arg=-fuse-ld=mold-extra
        run: cargo test
";

const COMMENTED_FLAGS: &str = "\
jobs:
  build:
    steps:
      - name: Test
        env:
          RUSTFLAGS: -D warnings # -Clink-arg=-fuse-ld=mold
        run: cargo test
";

const SIBLING_BORROWED: &str = "\
jobs:
  build:
    steps:
      - name: Test
        run: cargo test
      - name: Other
        env:
          RUSTFLAGS: -D warnings -Clink-arg=-fuse-ld=mold
        run: echo done
";

const PROSE_ONLY: &str = "\
jobs:
  build:
    steps:
      - name: Run cargo test later
        # cargo test is run by the next job
        run: echo skipped
";

const FOLDED_AFTER_ANCHOR: &str = "\
jobs:
  build:
    steps:
      - &suite
        name: Suite
        env:
          RUSTFLAGS: -D warnings -Clink-arg=-fuse-ld=mold
        run: |
          set -o pipefail
          cargo nextest run
      - *suite
";

/// Returns the steps of a fixture that failed, as a result a test can return.
fn expect_problems(text: &str, wanted: usize) -> Result<(), String> {
    let found = problems_in("fixture.yml", text);
    if found.len() == wanted {
        Ok(())
    } else {
        Err(format!("wanted {wanted} complaint(s), found {found:#?}"))
    }
}

#[test]
fn the_listed_workflows_hold_the_contract() -> Result<(), String> {
    let problems: Vec<String> = LISTED.iter().flat_map(listed_problems).collect();
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{problems:#?}"))
    }
}

#[test]
fn a_step_that_assigns_both_flags_is_accepted() -> Result<(), String> {
    expect_problems(COMPLIANT, 0)?;
    expect_problems(FOLDED_AFTER_ANCHOR, 0)
}

#[test]
fn a_step_that_assigns_nothing_or_too_little_is_refused() -> Result<(), String> {
    expect_problems(NO_ENV, 1)?;
    expect_problems(LACKS_LINKER, 1)?;
    expect_problems(LACKS_DENY, 1)?;
    expect_problems(LOOKALIKE_DENY, 1)?;
    expect_problems(COMMENTED_FLAGS, 1)
}

#[test]
fn a_step_cannot_borrow_a_siblings_assignment() -> Result<(), String> {
    expect_problems(SIBLING_BORROWED, 1)
}

#[test]
fn prose_naming_cargo_is_neither_a_step_nor_a_pass() {
    assert_eq!(cargo_step_count(PROSE_ONLY), 0);
    assert_eq!(cargo_step_count(NO_ENV), 1);
}

#[test]
fn a_changed_step_count_is_a_complaint() {
    let listed = Listed {
        file: "fixture.yml",
        text: COMPLIANT,
        cargo_steps: 2,
    };
    assert_eq!(listed_problems(&listed).len(), 1);
}

/// What one generated step holds: whether it runs `cargo`, and which
/// `RUSTFLAGS` it assigns.
#[derive(Clone, Copy)]
enum Shape {
    Complying,
    Unassigned,
    NoLinker,
    OnlyAssigns,
}

const SHAPES: [Shape; 4] = [
    Shape::Complying,
    Shape::Unassigned,
    Shape::NoLinker,
    Shape::OnlyAssigns,
];

impl Shape {
    /// Returns whether the step runs `cargo`.
    const fn runs_cargo(self) -> bool {
        !matches!(self, Self::OnlyAssigns)
    }

    /// Returns whether the step is one the contract accepts.
    const fn is_accepted(self) -> bool {
        matches!(self, Self::Complying | Self::OnlyAssigns)
    }

    /// Returns the step's text at an indentation, with a comment and a blank
    /// line before the next item.
    fn render(self, index: usize, margin: usize) -> String {
        let pad = " ".repeat(margin);
        let assignment = match self {
            Self::Complying | Self::OnlyAssigns => {
                format!("{pad}  env:\n{pad}    RUSTFLAGS: -D warnings {LINKER_FLAG}\n")
            }
            Self::NoLinker => format!("{pad}  env:\n{pad}    RUSTFLAGS: -D warnings\n"),
            Self::Unassigned => String::new(),
        };
        let run = if self.runs_cargo() {
            "cargo test"
        } else {
            "echo done"
        };
        format!(
            "{pad}- name: Step {index}\n{assignment}{pad}  run: {run}\n{pad}# between steps\n\n"
        )
    }
}

/// Returns a workflow of the given steps at an indentation.
fn workflow_of(shapes: &[Shape], margin: usize) -> String {
    let steps: String = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| shape.render(index, margin))
        .collect();
    format!("jobs:\n  build:\n    steps:\n{steps}")
}

/// Returns every ordered arrangement of `len` generated steps.
fn arrangements(len: usize) -> Vec<Vec<Shape>> {
    (0..len).fold(vec![Vec::new()], |prefixes, _| {
        prefixes
            .iter()
            .flat_map(|prefix| {
                SHAPES.iter().map(move |shape| {
                    let mut next = prefix.clone();
                    next.push(*shape);
                    next
                })
            })
            .collect()
    })
}

#[test]
fn every_arrangement_of_up_to_three_steps_is_judged_one_step_at_a_time() -> Result<(), String> {
    for margin in [6, 8] {
        for shapes in (1..=3).flat_map(arrangements) {
            let text = workflow_of(&shapes, margin);
            let cargo = shapes.iter().filter(|shape| shape.runs_cargo()).count();
            let refused = shapes
                .iter()
                .filter(|shape| shape.runs_cargo() && !shape.is_accepted())
                .count();
            let found_steps = cargo_step_count(&text);
            let found_problems = problems_in("generated.yml", &text).len();
            if found_steps != cargo || found_problems != refused {
                return Err(format!(
                    "margin {margin}: found {found_steps} cargo step(s) and {found_problems} \
                     complaint(s), wanted {cargo} and {refused}\n{text}"
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn the_step_the_cache_publisher_warms_assigns_the_flags_the_reader_does() -> Result<(), String> {
    let reader = rustflags_of_step(CI.text, WARMED_STEP);
    let writer = rustflags_of_step(COVERAGE_MAIN.text, WARMED_STEP);
    match (reader, writer) {
        (Some(read), Some(written)) if read == written => Ok(()),
        other => Err(format!(
            "`{WARMED_STEP}` must assign the same RUSTFLAGS in ci.yml and coverage-main.yml, \
             because the compiler cache entry is keyed on it; found {other:?}"
        )),
    }
}

#[test]
fn a_step_that_differs_from_its_cache_publisher_is_refused() {
    let reader = COMPLIANT.replace("Test", WARMED_STEP);
    let writer = reader.replace("-Clink-arg=-fuse-ld=mold", "-Clink-arg=-fuse-ld=lld");
    assert_ne!(
        rustflags_of_step(&reader, WARMED_STEP),
        rustflags_of_step(&writer, WARMED_STEP)
    );
    assert_eq!(rustflags_of_step(&reader, "Another step"), None);
}
