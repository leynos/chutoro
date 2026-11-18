//! Test-only counter for candidate-edge extractions.
//!
//! `extract_candidate_edges` records every call here so a regression test can
//! prove the non-harvesting insertion path never reaches it. Asserting on what
//! a collector receives cannot show that: a change could hoist the extraction
//! above the `if C::WANTS_EDGES` guard and still leave `collect` gated, which
//! reinstates the per-insert allocation the guard exists to remove.
//!
//! The counter is thread-local because `cargo test` runs sibling tests on
//! separate threads of one process, and only the inserting thread performs
//! extraction; a process-wide counter would let an unrelated test perturb the
//! observation. `cargo nextest`, which the repository gate uses, isolates each
//! test in its own process either way.

use std::cell::Cell;

thread_local! {
    static EXTRACTIONS: Cell<usize> = const { Cell::new(0) };
}

/// Records one extraction on the calling thread.
pub(crate) fn record() {
    EXTRACTIONS.with(|count| count.set(count.get() + 1));
}

/// Returns how many extractions the calling thread has performed.
pub(crate) fn count() -> usize {
    EXTRACTIONS.with(Cell::get)
}
