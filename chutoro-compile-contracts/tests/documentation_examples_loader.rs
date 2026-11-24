//! Failure contracts for the documented-example fence loader.

mod documentation_examples;

use documentation_examples::{
    DOCUMENTS, documented_example, load_documented_examples, parse_document,
};
use rstest::rstest;

#[rstest]
#[case::empty_identifier(
    "<!-- tested-example: -->\n",
    "tested-example identifier must not be empty"
)]
#[case::blank_identifier(
    "<!-- tested-example:  -->\n",
    "tested-example identifier must not be empty"
)]
#[case::malformed_marker("<!-- tested-example: open\n", "malformed tested-example marker")]
#[case::unmarked_backticks("```toml\n[dependencies]\n```\n", "missing a tested-example marker")]
#[case::unmarked_tildes("~~~toml\n[dependencies]\n~~~\n", "missing a tested-example marker")]
#[case::unterminated(
    "<!-- tested-example: sample -->\n```toml\n[dependencies]\n",
    "fence is not terminated"
)]
#[case::marker_without_fence("<!-- tested-example: sample -->\n", "marker has no fence")]
#[case::marker_before_prose(
    "<!-- tested-example: sample -->\nSome prose.\n",
    "expected an opening fence after marker"
)]
#[case::missing_language(
    "<!-- tested-example: sample -->\n```\nplain\n```\n",
    "fence should declare a language"
)]
#[case::duplicate(
    concat!(
        "<!-- tested-example: repeated -->\n```sh\nmake\n```\n",
        "<!-- tested-example: repeated -->\n```sh\nmake\n```\n"
    ),
    "duplicate tested-example identifier"
)]
fn malformed_documents_are_rejected(#[case] contents: &str, #[case] expected_message: &str) {
    let error = parse_document("fixture.md", contents)
        .expect_err("the malformed document should be rejected");
    assert!(
        error.to_string().contains(expected_message),
        "expected '{expected_message}' in '{error}'"
    );
}

#[test]
fn marked_fences_keep_their_identifier_language_and_body() {
    let contents = concat!(
        "Prose before.\n\n",
        "<!-- tested-example: kept -->\n\n",
        "```rust\nlet answer = 42;\n```\n",
    );
    let examples = parse_document("fixture.md", contents).expect("the document should load");
    let [example] = examples.as_slice() else {
        panic!("expected one example, got {examples:?}");
    };
    assert_eq!(example.id, "kept");
    assert_eq!(example.source, "fixture.md");
    assert_eq!(example.language, "rust");
    assert_eq!(example.body, "let answer = 42;\n");
}

/// A longer fence quotes a shorter one, so a marker-like line inside it is
/// body text and the inner fence does not close it.
#[test]
fn a_longer_fence_quotes_markers_and_shorter_fences() {
    let contents = concat!(
        "<!-- tested-example: quoting -->\n",
        "````text\n",
        "<!-- tested-example: quoted -->\n",
        "```sh\nmake\n```\n",
        "````\n",
    );
    let examples = parse_document("fixture.md", contents).expect("the document should load");
    let ids: Vec<&str> = examples.iter().map(|example| example.id.as_str()).collect();
    assert_eq!(ids, ["quoting"]);
    assert!(
        examples
            .iter()
            .all(|example| example.body.contains("```sh"))
    );
}

/// The published documents load, and every one of them contributes examples.
#[test]
fn the_published_documents_load() {
    let examples = load_documented_examples().expect("the published documents should load");
    for document in DOCUMENTS {
        assert!(
            examples.iter().any(|example| example.source == *document),
            "{document} should contain at least one example"
        );
    }
    let usage =
        documented_example("readme-library-usage").expect("the README usage example should load");
    assert_eq!(usage.language, "rust");
}
