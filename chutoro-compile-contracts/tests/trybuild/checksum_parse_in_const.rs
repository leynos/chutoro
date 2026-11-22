//! Compile-fail fixture: `Checksum::parse` is not a `const fn`.

use chutoro_bench_datasets::{Checksum, RecipeError};

const PARSED: Result<Checksum, RecipeError> = Checksum::parse(concat!(
    "sha256:",
    "0000000000000000000000000000000000000000000000000000000000000000",
));

fn main() {
    let _ = PARSED;
}