//! Compile-fail fixture: `HnswErrorCode` cannot be matched exhaustively.

use chutoro_core::HnswErrorCode;

fn describe(code: HnswErrorCode) -> &'static str {
    match code {
        HnswErrorCode::EmptyBuild => "empty build",
        HnswErrorCode::InvalidParameters => "invalid parameters",
        HnswErrorCode::DuplicateNode => "duplicate node",
        HnswErrorCode::GraphEmpty => "graph empty",
        HnswErrorCode::GraphInvariantViolation => "graph invariant violation",
        HnswErrorCode::NonFiniteDistance => "non-finite distance",
        HnswErrorCode::LockPoisoned => "poisoned lock",
        HnswErrorCode::DataSource => "data source failure",
    }
}

fn main() {
    let _ = describe(HnswErrorCode::EmptyBuild);
}