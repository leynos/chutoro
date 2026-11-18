//! Compile-fail fixture: an error's generated `code()` is `#[must_use]`.

#![deny(unused_must_use)]

use chutoro_core::MstError;

fn main() {
    MstError::EmptyGraph.code();
}