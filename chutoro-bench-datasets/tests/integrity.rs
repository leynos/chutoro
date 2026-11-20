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

#[test]
fn checksum_parse_round_trips_canonical_lowercase() {
    let text = format!("sha256:{CANONICAL_HEX}");
    let checksum = Checksum::parse(&text).expect("canonical checksum should parse");

    assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
    assert_eq!(checksum.to_string(), text);
}

#[test]
fn checksum_parse_normalizes_uppercase_and_compares_equal() {
    let lowercase = Checksum::parse(&format!("sha256:{CANONICAL_HEX}"))
        .expect("lowercase checksum should parse");
    let uppercase_text = format!("sha256:{}", CANONICAL_HEX.to_uppercase());
    let uppercase = Checksum::parse(&uppercase_text).expect("uppercase checksum should parse");

    assert_eq!(uppercase, lowercase);
    assert_eq!(uppercase.to_string(), format!("sha256:{CANONICAL_HEX}"));
}

#[test]
fn checksum_parse_accepts_mixed_case_digits() {
    let mixed: String = CANONICAL_HEX
        .chars()
        .enumerate()
        .map(|(index, character)| {
            if index.is_multiple_of(2) {
                character.to_ascii_uppercase()
            } else {
                character
            }
        })
        .collect();
    let checksum =
        Checksum::parse(&format!("sha256:{mixed}")).expect("mixed-case checksum should parse");

    assert_eq!(checksum.to_string(), format!("sha256:{CANONICAL_HEX}"));
}

#[test]
fn checksum_parse_rejects_bare_digest_without_prefix() {
    let failure = integrity_failure(expect_err!(
        Checksum::parse(CANONICAL_HEX),
        "a bare digest has no algorithm prefix"
    ));

    match failure {
        IntegrityFailure::MissingAlgorithmPrefix => {}
        other => panic!("expected MissingAlgorithmPrefix, got: {other:?}"),
    }
}

#[test]
fn checksum_parse_rejects_empty_prefix() {
    let text = format!(":{CANONICAL_HEX}");
    let failure = integrity_failure(expect_err!(
        Checksum::parse(&text),
        "an empty prefix should be rejected"
    ));

    match failure {
        IntegrityFailure::MissingAlgorithmPrefix => {}
        other => panic!("expected MissingAlgorithmPrefix, got: {other:?}"),
    }
}

#[rstest]
#[case::md5("md5")]
#[case::uppercase_sha256("SHA256")]
fn checksum_parse_rejects_unknown_algorithm(#[case] prefix: &str) {
    let text = format!("{prefix}:{CANONICAL_HEX}");
    let failure = integrity_failure(expect_err!(
        Checksum::parse(&text),
        "an unsupported prefix should be rejected"
    ));

    match failure {
        IntegrityFailure::UnknownAlgorithm { prefix: rejected } => {
            assert_eq!(rejected.as_ref(), prefix);
        }
        other => panic!("expected UnknownAlgorithm, got: {other:?}"),
    }
}

#[rstest]
#[case::empty(0)]
#[case::short(63)]
#[case::long(65)]
fn checksum_parse_rejects_wrong_digest_length(#[case] length: usize) {
    let payload = "a".repeat(length);
    let text = format!("sha256:{payload}");
    let failure = integrity_failure(expect_err!(
        Checksum::parse(&text),
        "a digest of the wrong width should be rejected"
    ));

    match failure {
        IntegrityFailure::InvalidDigestLength {
            algorithm,
            expected,
            actual,
        } => {
            assert_eq!(algorithm, DigestAlgorithm::Sha256);
            assert_eq!(expected, 64);
            assert_eq!(actual, length);
        }
        other => panic!("expected InvalidDigestLength, got: {other:?}"),
    }
}

#[test]
fn checksum_parse_reports_first_invalid_hex_position() {
    let payload = format!("{}g", "a".repeat(63));
    let text = format!("sha256:{payload}");
    let failure = integrity_failure(expect_err!(
        Checksum::parse(&text),
        "a non-hexadecimal digit should be rejected"
    ));

    match failure {
        IntegrityFailure::InvalidHexCharacter { position } => assert_eq!(position, 63),
        other => panic!("expected InvalidHexCharacter, got: {other:?}"),
    }
}

#[test]
fn checksum_parse_classifies_multibyte_payload_after_width_check() {
    // 62 hex digits plus a two-byte 'é' is 64 bytes, so the width check passes
    // and hexadecimal decoding reports the character's first byte.
    let payload = format!("{}é", "a".repeat(62));
    let text = format!("sha256:{payload}");
    let failure = integrity_failure(expect_err!(
        Checksum::parse(&text),
        "a multi-byte digest character should be rejected as non-hexadecimal"
    ));

    match failure {
        IntegrityFailure::InvalidHexCharacter { position } => assert_eq!(position, 62),
        other => panic!("expected InvalidHexCharacter, got: {other:?}"),
    }
}

#[test]
fn checksum_parse_rejects_multibyte_payload_before_decoding() {
    // 63 hex digits plus a two-byte 'é' is 65 bytes, so the width check wins
    // before any byte is decoded.
    let payload = format!("{}é", "a".repeat(63));
    let text = format!("sha256:{payload}");
    let failure = integrity_failure(expect_err!(
        Checksum::parse(&text),
        "a multi-byte payload of the wrong width should be rejected"
    ));

    match failure {
        IntegrityFailure::InvalidDigestLength { actual, .. } => assert_eq!(actual, 65),
        other => panic!("expected InvalidDigestLength, got: {other:?}"),
    }
}

#[test]
fn checksum_exposes_algorithm_and_digest_bytes() {
    let checksum = Checksum::parse(&format!("sha256:{CANONICAL_HEX}"))
        .expect("canonical checksum should parse");
    let leading: Vec<u8> = checksum.as_bytes().iter().copied().take(4).collect();

    assert_eq!(checksum.algorithm(), DigestAlgorithm::Sha256);
    assert_eq!(leading, vec![0x01, 0x23, 0x45, 0x67]);
}
