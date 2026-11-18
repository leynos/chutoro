//! Compile-fail fixture: a code enum's `as_str()` is `#[must_use]`.

#![deny(unused_must_use)]

use chutoro_core::MstErrorCode;

fn main() {
    MstErrorCode::EmptyGraph.as_str();
}