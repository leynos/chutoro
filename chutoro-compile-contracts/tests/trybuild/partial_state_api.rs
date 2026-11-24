//! Compile-pass fixture for the retained `PartialState` surface.

use chutoro_bench_datasets::{PartialState, Phase};

fn main() {
    let state = PartialState::new(Some(Phase::Validate));
    assert_eq!(state.highest_completed_phase, Some(Phase::Validate));

    let empty = PartialState::new(None);
    assert_eq!(empty.highest_completed_phase, None);
}