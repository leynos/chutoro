//! Compile-fail contract: `CleanupFailure` cannot be built outside its crate.

use chutoro_bench_datasets::{CleanupFailure, PortName, RecipeError};

fn main() {
    let _ = CleanupFailure {
        phase_error: RecipeError::prepare("archive corrupt"),
        cleanup_error: RecipeError::port(PortName::Storage, "cache locked"),
    };
}