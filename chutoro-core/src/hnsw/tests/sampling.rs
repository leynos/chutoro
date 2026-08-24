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
        let count = counts
            .get_mut(level)
            .expect("sampled level must be within the configured maximum");
        *count += 1;
    }

    let connection_count = u32::try_from(params.max_connections())
        .expect("test parameters fit within an exact f64 conversion");
    let continue_prob = f64::from(connection_count).recip();
    for window in counts
        .windows(2)
        .filter(|pair| matches!(pair, [current, next] if *current > 0 && *next > 0))
        .take(3)
    {
        let [current, next] = window else {
            continue;
        };
        let current_count =
            u32::try_from(*current).expect("sample count fits within an exact f64 conversion");
        let next_count =
            u32::try_from(*next).expect("sample count fits within an exact f64 conversion");
        let next_ratio = std::ops::Div::div(f64::from(next_count), f64::from(current_count));
        let ratio_error = std::ops::Sub::sub(next_ratio, continue_prob).abs();
        assert!(
            ratio_error < 0.035,
            "ratio should approach geometric tail (observed {next_ratio}, expected {continue_prob})",
        );
    }
}
