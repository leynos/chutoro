//! Compile-fail fixture: `MstErrorCode` cannot be matched exhaustively.

use chutoro_core::MstErrorCode;

fn describe(code: MstErrorCode) -> &'static str {
    match code {
        MstErrorCode::EmptyGraph => "empty graph",
        MstErrorCode::InvalidNodeId => "invalid node id",
        MstErrorCode::NonFiniteWeight => "non-finite weight",
        MstErrorCode::LockPoisoned => "poisoned lock",
        MstErrorCode::InvariantViolation => "invariant violation",
    }
}

fn main() {
    let _ = describe(MstErrorCode::EmptyGraph);
}