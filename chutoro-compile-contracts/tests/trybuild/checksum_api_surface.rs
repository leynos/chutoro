//! Compile-pass fixture: inhabited `Checksum` and boxed `IntegrityFailure`.
//!
//! The const items prove the constructor, the algorithm accessors, and the
//! checksum accessors are usable in const contexts.

use chutoro_bench_datasets::{Checksum, DigestAlgorithm, IntegrityFailure, RecipeError};

const ALGORITHM: DigestAlgorithm = DigestAlgorithm::Sha256;
const PREFIX: &str = DigestAlgorithm::Sha256.prefix();
const DIGEST_LEN: usize = DigestAlgorithm::Sha256.digest_len();
const CHECKSUM: Checksum = Checksum::new(ALGORITHM, [0; 32]);
const CHECKSUM_ALGORITHM: DigestAlgorithm = CHECKSUM.algorithm();
static CHECKSUM_INSTANCE: Checksum = CHECKSUM;
const CHECKSUM_BYTES: &[u8; 32] = CHECKSUM_INSTANCE.as_bytes();

fn describe(failure: &IntegrityFailure) -> &'static str {
    match failure {
        IntegrityFailure::MissingAlgorithmPrefix => "missing prefix",
        IntegrityFailure::UnknownAlgorithm { .. } => "unknown algorithm",
        IntegrityFailure::InvalidDigestLength { .. } => "invalid length",
        IntegrityFailure::InvalidHexCharacter { .. } => "invalid hex",
        _ => "other integrity failure",
    }
}

fn main() {
    let _ = PREFIX;
    let _ = DIGEST_LEN;
    let _ = CHECKSUM_ALGORITHM;
    let _ = CHECKSUM_BYTES;
    let _ = CHECKSUM.to_string();

    if let Ok(parsed) = Checksum::parse(&CHECKSUM.to_string()) {
        let _ = parsed.as_bytes();
    }

    let error = RecipeError::integrity(IntegrityFailure::MissingAlgorithmPrefix);
    if let RecipeError::Integrity(failure) = &error {
        let _ = describe(failure);
    }
}