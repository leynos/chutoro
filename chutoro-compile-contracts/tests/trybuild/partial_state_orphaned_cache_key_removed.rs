//! Compile-fail fixture proving the removed cleanup hint stays absent.

use chutoro_bench_datasets::{PartialState, Phase};

fn main() {
    let state = PartialState::new(Some(Phase::Validate));
    let _ = state.orphaned_cache_key;
}