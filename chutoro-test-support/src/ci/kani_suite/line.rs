//! Recognize and normalize one line of Kani's regular output.
//!
//! [`super::output`] keeps the state of one run; this module answers the
//! per-line questions that state asks: whether the line starts a harness
//! block, which outcome a verdict names, what the manual summary reports, and
//! what the line reads once its ANSI colour is removed.

use super::output::{Outcome, SuiteReport, Summary};

/// Return the harness a `Checking harness` line names, if it is one.
pub(super) fn checking_harness(line: &str) -> Option<String> {
    let name = line.strip_prefix("Checking harness ")?;
    Some(name.strip_suffix("...").unwrap_or(name).to_owned())
}

/// Map Kani's status text to an outcome.
pub(super) fn outcome_of(status: &str) -> Outcome {
    if status.starts_with("SUCCESSFUL") {
        Outcome::Success
    } else if status.starts_with("FAILED") {
        Outcome::Failed
    } else {
        Outcome::Other
    }
}

/// Read one line of the manual harness summary into `report`.
pub(super) fn read_summary_line(line: &str, report: &mut SuiteReport) {
    if let Some(name) = line.strip_prefix("Verification failed for - ") {
        report.reported_failures.push(name.trim().to_owned());
        return;
    }
    let Some(rest) = line.strip_prefix("Complete - ") else {
        return;
    };
    let numbers = rest
        .split_whitespace()
        .filter_map(|word| word.parse::<usize>().ok())
        .collect::<Vec<_>>();
    if let [succeeded, failed, total] = numbers.as_slice() {
        report.summary = Some(Summary {
            succeeded: *succeeded,
            failed: *failed,
            total: *total,
        });
    }
}

/// Return `text` with ANSI escape sequences removed.
///
/// Kani colours its verdict when it believes it is writing to a terminal. The
/// checker captures a pipe, so the colour is normally absent, but stripping it
/// unconditionally keeps a coloured run parseable rather than silently
/// unrecognised.
pub(super) fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '\u{1b}' {
            out.push(character);
            continue;
        }
        skip_escape(&mut characters);
    }
    out
}

/// Consume one ANSI escape sequence, leaving the iterator after its final byte.
///
/// Extracted from [`strip_ansi`] so the loop and its guards fit the nesting
/// threshold: the sequence body is one decision, not a fourth level of the
/// caller's loop.
fn skip_escape(characters: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    if characters.peek() != Some(&'[') {
        return;
    }
    characters.next();
    for escaped in characters.by_ref() {
        if escaped.is_ascii_alphabetic() {
            break;
        }
    }
}
