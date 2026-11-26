//! Rejection tests for the shared tested-example loader.
//!
//! `build.rs` uses the same `build_support/users_guide_examples.rs` module to
//! generate the compiled form of the published example. These cases pin each
//! condition the marker contract must reject, so a malformed document fails the
//! build rather than silently selecting the wrong text.

#[path = "../build_support/users_guide_examples.rs"]
mod users_guide_examples;

use users_guide_examples::{ExampleError, extract_example};

/// The published document the build script reads.
const EXAMPLE_DOCUMENT: &str = include_str!("../docs/session-refresh-example.md");

#[test]
fn the_guide_publishes_the_session_refresh_example() {
    let body = extract_example(EXAMPLE_DOCUMENT, "session-refresh")
        .expect("the document must declare the marked refresh example");

    assert!(
        body.contains("fn main() -> Result<(), ChutoroError>"),
        "the extracted example must be the complete program, got:\n{body}"
    );
    assert!(
        body.contains("session.refresh()?;"),
        "the extracted example must call refresh"
    );
    assert!(
        !body.contains("```"),
        "the fence lines must not be part of the extracted body"
    );
}

#[test]
fn an_unmarked_fence_is_rejected() {
    let markdown = concat!(
        "<!-- tested-example: marked -->\n",
        "```rust\nfn main() {}\n```\n",
        "```rust\nfn main() {}\n```\n",
    );

    assert_eq!(
        extract_example(markdown, "marked"),
        Err(ExampleError::UnmarkedFence { line: 5 })
    );
}

#[test]
fn a_marked_document_can_lack_the_requested_identifier() {
    let markdown = "<!-- tested-example: present -->\n```rust\nfn main() {}\n```\n";

    assert_eq!(
        extract_example(markdown, "absent"),
        Err(ExampleError::MissingExample {
            id: String::from("absent"),
        })
    );
}

#[test]
fn an_unterminated_fence_is_rejected() {
    let markdown = "<!-- tested-example: broken -->\n```rust\nfn main() {}\n";

    assert_eq!(
        extract_example(markdown, "broken"),
        Err(ExampleError::UnterminatedFence { line: 2 })
    );
}

#[test]
fn a_duplicate_identifier_is_rejected() {
    let markdown = concat!(
        "<!-- tested-example: twice -->\n",
        "```rust\nfn main() {}\n```\n",
        "<!-- tested-example: twice -->\n",
        "```rust\nfn main() {}\n```\n",
    );

    assert_eq!(
        extract_example(markdown, "twice"),
        Err(ExampleError::DuplicateIdentifier {
            id: String::from("twice"),
            line: 5,
        })
    );
}

#[test]
fn a_marker_without_a_fence_is_rejected() {
    let markdown = "<!-- tested-example: detached -->\nprose instead of a fence\n";

    assert_eq!(
        extract_example(markdown, "detached"),
        Err(ExampleError::MarkerWithoutFence { line: 1 })
    );
}

#[test]
fn a_marker_without_an_identifier_is_rejected() {
    let markdown = "<!-- tested-example: -->\n```rust\nfn main() {}\n```\n";

    assert_eq!(
        extract_example(markdown, "empty"),
        Err(ExampleError::MalformedMarker { line: 1 })
    );
}
