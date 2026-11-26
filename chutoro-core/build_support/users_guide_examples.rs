//! Extracts marker-selected fenced examples from a published Markdown guide.
//!
//! The users' guide is the source of truth for its worked examples. A marker
//! comment of the form `<!-- tested-example: <identifier> -->` immediately
//! above an opening fence names that fence, and [`extract_example`] returns its
//! body so the build script can compile the shipped text instead of a copy.
//!
//! The loader rejects a malformed or empty identifier, a marker with no fence
//! beneath it, an unterminated fence, and a duplicate identifier. It is shared
//! by the build script and by `tests/users_guide_loader.rs`, so the rejection
//! rules are tested on the same code that generates the compiled example.

use std::{collections::BTreeSet, error::Error, fmt};

/// Opening text of a tested-example marker.
const MARKER_PREFIX: &str = "<!-- tested-example:";
/// Closing text of a tested-example marker.
const MARKER_SUFFIX: &str = "-->";
/// Opening text of a Markdown fenced code block.
const FENCE: &str = "```";

/// One marked example loaded from a document.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct MarkedExample {
    /// Stable identifier declared by the marker.
    pub(crate) id: String,
    /// Fence body, excluding the opening and closing fence lines.
    pub(crate) body: String,
}

/// Why marked examples could not be loaded from a document.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ExampleError {
    /// A marker could not be parsed or carried no identifier.
    MalformedMarker {
        /// One-based line number of the offending marker.
        line: usize,
    },
    /// A marker was not immediately followed by an opening fence.
    MarkerWithoutFence {
        /// One-based line number of the offending marker.
        line: usize,
    },
    /// A fence was opened and never closed.
    UnterminatedFence {
        /// One-based line number of the opening fence.
        line: usize,
    },
    /// Two markers declared the same identifier.
    DuplicateIdentifier {
        /// Identifier declared twice.
        id: String,
        /// One-based line number of the second declaration.
        line: usize,
    },
    /// The requested identifier is absent.
    MissingExample {
        /// Identifier that no marker declares.
        id: String,
    },
}

impl fmt::Display for ExampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedMarker { line } => {
                write!(formatter, "line {line}: malformed tested-example marker")
            }
            Self::MarkerWithoutFence { line } => {
                write!(
                    formatter,
                    "line {line}: tested-example marker is not followed by a fence"
                )
            }
            Self::UnterminatedFence { line } => {
                write!(formatter, "line {line}: fenced example is never closed")
            }
            Self::DuplicateIdentifier { id, line } => {
                write!(formatter, "line {line}: duplicate tested-example `{id}`")
            }
            Self::MissingExample { id } => {
                write!(formatter, "no tested-example marker declares `{id}`")
            }
        }
    }
}

impl Error for ExampleError {}

/// Returns the body of the example identified by `id`.
///
/// # Errors
///
/// Returns [`ExampleError`] when the document's markers are malformed,
/// duplicated, detached from a fence or unterminated, and when no marker
/// declares `id`. An unmarked fence is therefore invisible: the only way to
/// load an example is to mark it.
pub(crate) fn extract_example(markdown: &str, id: &str) -> Result<String, ExampleError> {
    load_examples(markdown)?
        .into_iter()
        .find(|example| example.id == id)
        .map(|example| example.body)
        .ok_or_else(|| ExampleError::MissingExample { id: id.to_owned() })
}

/// One source line, paired with its one-based number.
#[derive(Clone, Copy)]
struct SourceLine<'a> {
    /// The line's text, without its terminator.
    text: &'a str,
    /// One-based line number, used in diagnostics.
    number: usize,
}

impl<'a> SourceLine<'a> {
    /// Pairs `text` with its one-based line number.
    const fn new(text: &'a str, number: usize) -> Self {
        Self { text, number }
    }
}

/// Parses every marked example, enforcing the marker contract.
fn load_examples(markdown: &str) -> Result<Vec<MarkedExample>, ExampleError> {
    let mut loader = Loader::default();
    for (offset, text) in markdown.lines().enumerate() {
        loader.accept(SourceLine::new(text, offset.saturating_add(1)))?;
    }
    loader.finish()
}

/// A fence currently being read.
#[derive(Default)]
struct Loader {
    /// Examples whose closing fence has been seen.
    examples: Vec<MarkedExample>,
    /// Identifiers seen so far, for duplicate detection.
    identifiers: BTreeSet<String>,
    /// Marker awaiting its opening fence.
    pending: Option<(String, usize)>,
    /// Fence currently being read, if any.
    open: Option<OpenFence>,
}

impl Loader {
    /// Consumes one line, returning the first contract violation.
    fn accept(&mut self, line: SourceLine<'_>) -> Result<(), ExampleError> {
        if self.open.is_some() {
            self.accept_inside_fence(line);
            return Ok(());
        }
        if line.text.trim().starts_with(MARKER_PREFIX) {
            return self.accept_marker(line);
        }
        if is_opening_fence(line) {
            self.open = Some(OpenFence {
                id: self.pending.take().map(|(identifier, _line)| identifier),
                line: line.number,
                body: String::new(),
            });
            return Ok(());
        }
        self.require_no_pending()
    }

    /// Consumes a line while a fence is open, closing the fence when it ends.
    fn accept_inside_fence(&mut self, line: SourceLine<'_>) {
        if !is_closing_fence(line) {
            if let Some(fence) = self.open.as_mut() {
                fence.body.push_str(line.text);
                fence.body.push('\n');
            }
            return;
        }
        if let Some(finished) = self.open.take()
            && let Some(id) = finished.id
        {
            self.examples.push(MarkedExample {
                id,
                body: finished.body,
            });
        }
    }

    /// Records a marker and rejects a detached or duplicate one.
    fn accept_marker(&mut self, line: SourceLine<'_>) -> Result<(), ExampleError> {
        if let Some((_, marker_line)) = self.pending.as_ref() {
            return Err(ExampleError::MarkerWithoutFence { line: *marker_line });
        }
        let identifier = parse_identifier(line)?;
        if !self.identifiers.insert(identifier.to_owned()) {
            return Err(ExampleError::DuplicateIdentifier {
                id: identifier.to_owned(),
                line: line.number,
            });
        }
        self.pending = Some((identifier.to_owned(), line.number));
        Ok(())
    }

    /// Rejects any line that follows a marker without being a fence.
    const fn require_no_pending(&self) -> Result<(), ExampleError> {
        match self.pending.as_ref() {
            Some((_, marker_line)) => Err(ExampleError::MarkerWithoutFence { line: *marker_line }),
            None => Ok(()),
        }
    }

    /// Checks the document ended with no open fence or detached marker.
    fn finish(self) -> Result<Vec<MarkedExample>, ExampleError> {
        if let Some(fence) = self.open {
            return Err(ExampleError::UnterminatedFence { line: fence.line });
        }
        if let Some((_, marker_line)) = self.pending {
            return Err(ExampleError::MarkerWithoutFence { line: marker_line });
        }
        Ok(self.examples)
    }
}

/// A fence currently being read.
struct OpenFence {
    /// Identifier from the preceding marker, or `None` for an unmarked fence.
    id: Option<String>,
    /// One-based line number of the opening fence.
    line: usize,
    /// Body accumulated so far.
    body: String,
}

/// Parses the identifier that follows a marker prefix.
fn parse_identifier(line: SourceLine<'_>) -> Result<&'_ str, ExampleError> {
    let rest = line.text.trim().trim_start_matches(MARKER_PREFIX);
    let Some(identifier) = rest.strip_suffix(MARKER_SUFFIX) else {
        return Err(ExampleError::MalformedMarker { line: line.number });
    };
    let trimmed = identifier.trim();
    if trimmed.is_empty() || trimmed.contains(char::is_whitespace) {
        return Err(ExampleError::MalformedMarker { line: line.number });
    }
    Ok(trimmed)
}

/// Reports whether `line` opens a fenced block, with or without a language.
fn is_opening_fence(line: SourceLine<'_>) -> bool {
    line.text.trim_start().starts_with(FENCE)
}

/// Reports whether `line` is a bare closing fence.
fn is_closing_fence(line: SourceLine<'_>) -> bool {
    line.text.trim() == FENCE
}
