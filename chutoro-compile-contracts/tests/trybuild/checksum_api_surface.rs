//! Compile-pass fixture: inhabited `Checksum` and boxed `IntegrityFailure`.

use chutoro_bench_datasets::{Checksum, DigestAlgorithm, IntegrityFailure, RecipeError};

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
    let checksum = Checksum::new(DigestAlgorithm::Sha256, [0; 32]);
    let _ = checksum.algorithm();
    let _ = checksum.as_bytes();
    let _ = checksum.to_string();

    if let Ok(parsed) = Checksum::parse(&checksum.to_string()) {
        let _ = parsed.as_bytes();
    }

    let error = RecipeError::integrity(IntegrityFailure::MissingAlgorithmPrefix);
    if let RecipeError::Integrity(failure) = &error {
        let _ = describe(failure);
    }
}