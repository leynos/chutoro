//! Loads fenced examples from user-facing Markdown for executable tests.
//!
//! Every fence in the documents listed in [`DOCUMENTS`] must be preceded by a
//! `tested-example` marker naming a stable identifier. Rust fences are
//! compiled and run as doctests through `chutoro-compile-contracts/src/lib.rs`;
//! every other fence is checked by a semantic contract keyed by its
//! identifier. Contract tests load the published text through this module
//! rather than keeping copied fixtures, so an edited example cannot drift from
//! its test.

use std::collections::HashSet;

use anyhow::{Context, Result, bail, ensure};
use camino::Utf8PathBuf;

/// User-facing documents whose fences are all tested, relative to the
/// repository root.
pub const DOCUMENTS: &[&str] = &[
    "README.md",
    "docs/users-guide.md",
    "docs/v0-1-0-migration-guide.md",
];

const MARKER_PREFIX: &str = "<!-- tested-example:";
const MARKER_SUFFIX: &str = "-->";

/// One marked fenced example loaded from a user-facing document.
#[derive(Debug, Eq, PartialEq)]
pub struct DocumentedExample {
    /// Stable identifier declared by the `tested-example` marker.
    pub id: String,
    /// Document the example was loaded from, relative to the repository root.
    pub source: &'static str,
    /// Markdown fence info string, such as `rust` or `toml`.
    pub language: String,
    /// Exact text inside the fence, including a trailing newline.
    pub body: String,
}

/// Position of a line in a document, for error messages.
#[derive(Clone, Copy)]
struct Cursor {
    source: &'static str,
    line_index: usize,
}

impl Cursor {
    fn error(self, message: &str) -> String {
        format!("{}:{} {message}", self.source, self.line_index + 1)
    }
}

/// Return the repository root, which holds the documented files.
pub fn repository_root() -> Utf8PathBuf {
    Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(|| Utf8PathBuf::from(".."), ToOwned::to_owned)
}

/// Read a repository file as UTF-8 text.
///
/// # Errors
///
/// Returns an error when the file cannot be read.
pub fn read_repository_file(path: &str) -> Result<String> {
    let full_path = repository_root().join(path);
    let directory = cap_std::fs_utf8::Dir::open_ambient_dir(
        full_path.parent().context("documented path has a parent")?,
        cap_std::ambient_authority(),
    )
    .with_context(|| format!("open the directory holding {path}"))?;
    directory
        .read_to_string(
            full_path
                .file_name()
                .context("documented path has a name")?,
        )
        .with_context(|| format!("read {path}"))
}

/// Load every marked example and reject unmarked or duplicate fences.
///
/// # Errors
///
/// Returns an error when a document cannot be read, a marker is malformed,
/// a fence is unmarked or unterminated, or an identifier is duplicated.
pub fn load_documented_examples() -> Result<Vec<DocumentedExample>> {
    let mut examples = Vec::new();
    for &path in DOCUMENTS {
        examples.extend(parse_document(path, &read_repository_file(path)?)?);
    }

    let mut ids = HashSet::new();
    for example in &examples {
        ensure!(
            ids.insert(example.id.as_str()),
            "duplicate tested-example identifier '{}'",
            example.id
        );
    }
    Ok(examples)
}

/// Load the documented example identified by `id`.
///
/// # Errors
///
/// Returns an error when the documents are invalid or `id` is absent.
pub fn documented_example(id: &str) -> Result<DocumentedExample> {
    load_documented_examples()?
        .into_iter()
        .find(|example| example.id == id)
        .with_context(|| format!("documented example '{id}' should exist"))
}

/// Parse one document, requiring a marker before every fence.
///
/// # Errors
///
/// Returns an error when a marker is malformed, a fence is unmarked or
/// unterminated, or an identifier is duplicated within the document.
pub fn parse_document(source: &'static str, contents: &str) -> Result<Vec<DocumentedExample>> {
    let mut lines = contents.lines().enumerate();
    let mut examples = Vec::new();
    let mut ids = HashSet::new();

    while let Some((line_index, line)) = lines.next() {
        let cursor = Cursor { source, line_index };
        if let Some(id) = parse_marker(line) {
            ensure!(
                !id.is_empty(),
                "{}",
                cursor.error("tested-example identifier must not be empty")
            );
            ensure!(ids.insert(id), "duplicate tested-example identifier '{id}'");
            examples.push(read_marked_example(cursor, id, &mut lines)?);
        } else {
            ensure!(
                !line.starts_with(MARKER_PREFIX),
                "{}",
                cursor.error("malformed tested-example marker")
            );
            ensure!(
                FenceOpening::parse(line).is_none(),
                "{}",
                cursor.error("fence is missing a tested-example marker")
            );
        }
    }

    Ok(examples)
}

fn read_marked_example<'a>(
    cursor: Cursor,
    id: &str,
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<DocumentedExample> {
    let (fence_index, fence) = lines
        .find(|(_, line)| !line.is_empty())
        .with_context(|| cursor.error("marker has no fence"))?;
    let fence_cursor = Cursor {
        source: cursor.source,
        line_index: fence_index,
    };
    let opening = FenceOpening::parse(fence)
        .with_context(|| fence_cursor.error("expected an opening fence after marker"))?;
    ensure!(
        !opening.info.is_empty(),
        "{}",
        fence_cursor.error("fence should declare a language")
    );
    let body = read_fence_body(fence_cursor, opening, lines)?;
    Ok(DocumentedExample {
        id: id.to_owned(),
        source: cursor.source,
        language: opening.info.to_owned(),
        body,
    })
}

/// Return the identifier of a marker line, or `None` for any other line.
///
/// The identifier is trimmed, so an empty marker yields `Some("")` and is
/// reported as empty rather than malformed.
fn parse_marker(line: &str) -> Option<&str> {
    line.strip_prefix(MARKER_PREFIX)
        .and_then(|value| value.strip_suffix(MARKER_SUFFIX))
        .map(str::trim)
}

/// The opening line of a fenced code block.
///
/// A fence is a run of at least three backticks or tildes at column 0, and it
/// closes only on a line of the same character at least as long, so a longer
/// fence can quote a shorter one.
#[derive(Clone, Copy)]
struct FenceOpening<'a> {
    delimiter: char,
    run_length: usize,
    info: &'a str,
}

impl<'a> FenceOpening<'a> {
    /// Recognize `line` as a fence opening, capturing its delimiter and run.
    fn parse(line: &'a str) -> Option<Self> {
        let delimiter = line.chars().next().filter(|ch| matches!(ch, '`' | '~'))?;
        let info = line.trim_start_matches(delimiter);
        // Both delimiters are one byte, so the byte difference is the run.
        let run_length = line.len().checked_sub(info.len())?;
        (run_length >= 3).then_some(Self {
            delimiter,
            run_length,
            info: info.trim(),
        })
    }

    /// Report whether `line` closes this fence.
    fn is_closed_by(self, line: &str) -> bool {
        let candidate = line.trim_end();
        candidate.len() >= self.run_length && candidate.chars().all(|ch| ch == self.delimiter)
    }
}

fn read_fence_body<'a>(
    cursor: Cursor,
    opening: FenceOpening<'_>,
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<String> {
    let mut body = String::new();
    for (_, line) in lines {
        if opening.is_closed_by(line) {
            return Ok(body);
        }
        body.push_str(line);
        body.push('\n');
    }
    bail!("{}", cursor.error("fence is not terminated"))
}
