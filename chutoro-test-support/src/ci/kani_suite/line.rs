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

/// The exact status text Kani prints after proving a harness.
const SUCCESSFUL: &str = "SUCCESSFUL";

/// The exact status text Kani prints after falsifying a harness.
const FAILED: &str = "FAILED";

/// Map exactly one of Kani's two verdict statuses to an outcome.
///
/// The match is exact rather than a prefix. A prefix match would accept a
/// changed or extended marker such as `SUCCESSFUL_BUT_CHANGED` as a proof, and
/// this checker exists to make a changed result surface fail rather than pass.
/// Anything unrecognized becomes [`Outcome::Other`], which the verdict refuses.
pub(super) fn outcome_of(status: &str) -> Outcome {
    match status {
        SUCCESSFUL => Outcome::Success,
        FAILED => Outcome::Failed,
        _ => Outcome::Other,
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
    if let Some(summary) = summary_of(rest) {
        report.summary = Some(summary);
    }
}

/// Parse the exact totals grammar Kani prints after `Complete - `.
///
/// A word-by-word scan of every integer in the line would accept a changed
/// marker that happens to carry three numbers, and would also accept the
/// numbers in a different order. The separators are matched literally, so a
/// format change leaves the summary absent; a run without a summary is then
/// refused by the verdict rather than read as complete.
fn summary_of(rest: &str) -> Option<Summary> {
    let (succeeded, after_succeeded) = rest.split_once(" successfully verified harnesses, ")?;
    let (failed, after_failed) = after_succeeded.split_once(" failures, ")?;
    let total = after_failed.strip_suffix(" total.")?;
    Some(Summary {
        succeeded: decimal(succeeded)?,
        failed: decimal(failed)?,
        total: decimal(total)?,
    })
}

/// Parse one unsigned decimal field, accepting digits only.
///
/// `str::parse` accepts a leading `+`, which Kani never prints, so the field is
/// checked as digits before conversion.
fn decimal(field: &str) -> Option<usize> {
    let is_decimal = !field.is_empty() && field.bytes().all(|byte| byte.is_ascii_digit());
    is_decimal.then(|| field.parse().ok()).flatten()
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
