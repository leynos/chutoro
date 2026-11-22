//! Compile-fail fixture: `RecipeError::ChecksumUnsupported` no longer exists.

use chutoro_bench_datasets::RecipeError;

fn is_unsupported(error: &RecipeError) -> bool {
    matches!(error, RecipeError::ChecksumUnsupported)
}

fn main() {
    let _ = is_unsupported;
}