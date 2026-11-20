//! Integration tests for the checksum parsing contract.

use chutoro_bench_datasets::{Checksum, DigestAlgorithm, IntegrityFailure, RecipeError};
use rstest::rstest;

mod common;
use common::expect_err;

/// Canonical lowercase 64-character hexadecimal digest payload.
const CANONICAL_HEX: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// Unwrap a checksum error into its typed integrity failure.
fn integrity_failure(error: RecipeError) -> IntegrityFailure {
    match error {
        RecipeError::Integrity(failure) => *failure,
        other => panic!("expected an integrity failure, got: {other}"),
    }
}

/// Build a `sha256:`-prefixed value from a digest payload.
fn checksum_text(payload: &str) -> String {
    format!("sha256:{payload}")
}

/// Require `text` to be rejected and return its typed integrity failure.
fn rejected(text: &str) -> IntegrityFailure {
    integrity_failure(expect_err!(
        Checksum::parse(text),
        "fixture should be rejected"
    ))
}

/// Upper-case every hexadecimal digit of `hex`.
fn upper_case(hex: &str) -> String {
    hex.to_uppercase()
}

/// Upper-case every even-indexed hexadecimal digit of `hex`.
fn mixed_case(hex: &str) -> String {
    hex.chars()
        .enumerate()
        .map(|(index, character)| {
            if index.is_multiple_of(2) {
                character.to_ascii_uppercase()
            } else {
                character
            }
        })
        .collect()
}

#[test]
fn checksum_parse_round_trips_canonical_lowercase() {
    let text = checksum_text(CANONICAL_HEX);
    let checksum = Checksum::parse(&text).expect("canonical checksum should parse");

    assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
    assert_eq!(checksum.to_string(), text);
}

#[rstest]
#[case::uppercase(upper_case as fn(&str) -> String)]
#[case::mixed(mixed_case as fn(&str) -> String)]
fn checksum_parse_normalizes_hex_case(#[case] transform: fn(&str) -> String) {
    let canonical =
        Checksum::parse(&checksum_text(CANONICAL_HEX)).expect("lowercase checksum should parse");
    let variant = Checksum::parse(&checksum_text(&transform(CANONICAL_HEX)))
        .expect("case variant should parse");

    assert_eq!(variant, canonical);
    assert_eq!(variant.to_string(), checksum_text(CANONICAL_HEX));
}

#[rstest]
#[case::bare_digest("")]
#[case::empty_prefix(":")]
fn checksum_parse_rejects_missing_algorithm_prefix(#[case] prefix: &str) {
    let failure = rejected(&format!("{prefix}{CANONICAL_HEX}"));

    assert!(
        matches!(failure, IntegrityFailure::MissingAlgorithmPrefix),
        "expected MissingAlgorithmPrefix, got: {failure:?}",
    );
}

#[rstest]
#[case::md5("md5")]
#[case::uppercase_sha256("SHA256")]
fn checksum_parse_rejects_unknown_algorithm(#[case] prefix: &str) {
    let failure = rejected(&format!("{prefix}:{CANONICAL_HEX}"));

    let IntegrityFailure::UnknownAlgorithm { prefix: reported } = failure else {
        panic!("expected UnknownAlgorithm, got: {failure:?}");
    };
    assert_eq!(reported.as_ref(), prefix);
}

#[rstest]
#[case::empty(0)]
#[case::short(63)]
#[case::long(65)]
fn checksum_parse_rejects_wrong_digest_length(#[case] length: usize) {
    let failure = rejected(&checksum_text(&"a".repeat(length)));

    let IntegrityFailure::InvalidDigestLength {
        algorithm,
        expected,
        actual,
    } = failure
    else {
        panic!("expected InvalidDigestLength, got: {failure:?}");
    };
    assert_eq!(algorithm, DigestAlgorithm::Sha256);
    assert_eq!(expected, 64);
    assert_eq!(actual, length);
}

#[test]
fn checksum_parse_reports_first_invalid_hex_position() {
    let failure = rejected(&checksum_text(&format!("{}g", "a".repeat(63))));

    let IntegrityFailure::InvalidHexCharacter { position } = failure else {
        panic!("expected InvalidHexCharacter, got: {failure:?}");
    };
    assert_eq!(position, 63);
}

#[test]
fn checksum_parse_classifies_multibyte_payload_after_width_check() {
    // 62 hex digits plus a two-byte 'é' is 64 bytes, so the width check passes
    // and hexadecimal decoding reports the character's first byte.
    let failure = rejected(&checksum_text(&format!("{}é", "a".repeat(62))));

    let IntegrityFailure::InvalidHexCharacter { position } = failure else {
        panic!("expected InvalidHexCharacter, got: {failure:?}");
    };
    assert_eq!(position, 62);
}

#[test]
fn checksum_parse_rejects_multibyte_payload_before_decoding() {
    // 63 hex digits plus a two-byte 'é' is 65 bytes, so the width check wins
    // before any byte is decoded.
    let failure = rejected(&checksum_text(&format!("{}é", "a".repeat(63))));

    let IntegrityFailure::InvalidDigestLength { actual, .. } = failure else {
        panic!("expected InvalidDigestLength, got: {failure:?}");
    };
    assert_eq!(actual, 65);
}

#[test]
fn checksum_exposes_algorithm_and_digest_bytes() {
    let checksum =
        Checksum::parse(&checksum_text(CANONICAL_HEX)).expect("canonical checksum should parse");
    let leading: Vec<u8> = checksum.as_bytes().iter().copied().take(4).collect();

    assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
    assert_eq!(leading, vec![0x01, 0x23, 0x45, 0x67]);
}
