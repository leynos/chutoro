//! Compile-fail fixture: `IntegrityFailure` cannot be matched exhaustively.

use chutoro_bench_datasets::IntegrityFailure;

fn describe(failure: IntegrityFailure) -> &'static str {
    match failure {
        IntegrityFailure::MissingAlgorithmPrefix => "missing prefix",
        IntegrityFailure::UnknownAlgorithm { .. } => "unknown algorithm",
        IntegrityFailure::InvalidDigestLength { .. } => "invalid length",
        IntegrityFailure::InvalidHexCharacter { .. } => "invalid hex",
    }
}

fn main() {
    let _ = describe(IntegrityFailure::MissingAlgorithmPrefix);
}