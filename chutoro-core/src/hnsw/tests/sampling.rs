//! Sampling tests ensuring level distribution matches the geometric tail.

use rand::{Rng, SeedableRng, distributions::Standard, rngs::SmallRng};

use crate::hnsw::HnswParams;

#[test]
fn level_sampling_matches_geometric_tail() {
    let params = HnswParams::new(16, 64)
        .expect("params must be valid")
        .with_rng_seed(1337);
    let mut rng = SmallRng::seed_from_u64(params.rng_seed());
    let mut counts = vec![0_usize; params.max_level() + 1];
    let samples = 10_000;
    for _ in 0..samples {
        let mut level = 0_usize;
        while level < params.max_level() {
            let draw: f64 = rng.sample(Standard);
            if params.should_stop(draw) {
                break;
            }
            level += 1;
        }
        counts[level] += 1;
    }

    let continue_prob = 1.0 / params.max_connections() as f64;
    for window in counts
        .windows(2)
        .filter(|pair| pair[0] > 0 && pair[1] > 0)
        .take(3)
    {
        let next_ratio = window[1] as f64 / window[0] as f64;
        assert!(
            (next_ratio - continue_prob).abs() < 0.035,
            "ratio should approach geometric tail (observed {next_ratio}, expected {continue_prob})",
        );
    }
}
