//! Compile-fail fixture: `DigestAlgorithm` cannot be matched exhaustively.

use chutoro_bench_datasets::DigestAlgorithm;

fn describe(algorithm: DigestAlgorithm) -> &'static str {
    match algorithm {
        DigestAlgorithm::Sha256 => "sha256",
    }
}

fn main() {
    let _ = describe(DigestAlgorithm::Sha256);
}