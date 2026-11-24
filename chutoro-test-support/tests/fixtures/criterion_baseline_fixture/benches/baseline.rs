//! Isolated Criterion fixture for the benchmark baseline end-to-end test.
//!
//! The fixture registers the identifier shapes the helper must handle: a
//! plain parameter, one containing a slash (which Criterion sanitizes to an
//! underscore in the directory name while keeping it in `full_id`), and one
//! containing `=`. The measured unit is a deterministic arithmetic loop scaled
//! by `CHUTORO_FIXTURE_SLOWDOWN`, so a test can create a controlled slowdown
//! without depending on machine speed.

// The repository's `.cargo/config.toml` sets `-D missing_docs` for every crate
// beneath it. `criterion_group!` expands to a generated public harness
// function, which cannot carry a doc comment, so this isolated fixture — which
// is a bench harness rather than a public API — relaxes that one lint.
#![allow(missing_docs)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

/// Run a deterministic arithmetic loop with the requested number of units.
fn work(units: u64) -> u64 {
    let mut value = 1_u64;
    for index in 1..=units {
        value = value
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(index | 1);
    }
    value
}

/// Return the multiplier requested by the test environment.
fn slowdown() -> u64 {
    std::env::var("CHUTORO_FIXTURE_SLOWDOWN")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

/// Register the fixture identifiers.
fn baseline(c: &mut Criterion) {
    let multiplier = slowdown();
    let mut group = c.benchmark_group("fixture");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(50));
    group.measurement_time(Duration::from_millis(250));

    for (name, parameter) in [
        ("alpha", "plain"),
        ("alpha", "with/slash"),
        ("beta", "n=1,eq=a"),
    ] {
        group.bench_with_input(
            BenchmarkId::new(name, parameter),
            &multiplier,
            |bencher, &scale| {
                bencher.iter(|| black_box(work(black_box(2_000 * scale))));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, baseline);
criterion_main!(benches);