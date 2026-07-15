//! Neighbour scoring Criterion entrypoint.
//!
//! Testable planning, profiling, and orchestration live in the benchmark
//! support library so Cargo's normal test harness executes their unit tests.

use chutoro_benches::neighbour_scoring::run_neighbour_scoring;
use criterion::{Criterion, criterion_main};

fn neighbour_scoring(c: &mut Criterion) {
    run_neighbour_scoring(c);
}

mod bench_harness {
    //! Criterion benchmark entrypoint for neighbour-scoring measurements.

    use super::neighbour_scoring;
    use criterion::criterion_group;

    criterion_group!(benches, neighbour_scoring);
}

criterion_main!(bench_harness::benches);
