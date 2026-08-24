//! RNG setup and level sampling utilities for the CPU HNSW implementation.

use std::sync::Mutex;

use rand::{Rng, SeedableRng, distributions::Standard, rngs::SmallRng};
use rayon::{current_num_threads, current_thread_index};

use crate::hnsw::error::HnswError;

use super::CpuHnsw;

/// `SplitMix64` increment (the 64-bit golden ratio) used for per-worker seed
/// derivation.
const WORKER_SEED_SPACING: u64 = 0x9E37_79B9_7F4A_7C15;
/// First `SplitMix64` multiplier used to diffuse worker-seed bits.
const SPLITMIX_MULT_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second `SplitMix64` multiplier used to diffuse worker-seed bits.
const SPLITMIX_MULT_B: u64 = 0x94D0_49BB_1331_11EB;

/// Derive a deterministic, distinct seed for one Rayon worker.
#[inline]
pub(super) const fn mix_worker_seed(base_seed: u64, worker_index: usize) -> u64 {
    splitmix64(base_seed ^ ((worker_index as u64 + 1).wrapping_mul(WORKER_SEED_SPACING)))
}

/// Mix one 64-bit state into a pseudo-random worker seed.
#[inline]
const fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(WORKER_SEED_SPACING);
    state = (state ^ (state >> 30)).wrapping_mul(SPLITMIX_MULT_A);
    state = (state ^ (state >> 27)).wrapping_mul(SPLITMIX_MULT_B);
    state ^ (state >> 31)
}

/// Build one independently seeded random-number generator per Rayon worker.
pub(super) fn build_worker_rngs(base_seed: u64) -> Vec<Mutex<SmallRng>> {
    (0..current_num_threads())
        .map(|idx| {
            let seed = mix_worker_seed(base_seed, idx);
            Mutex::new(SmallRng::seed_from_u64(seed))
        })
        .collect()
}

impl CpuHnsw {
    /// Sample a random HNSW level from the current worker's generator.
    pub(super) fn sample_level(&self) -> Result<usize, HnswError> {
        if let Some(index) = current_thread_index()
            && let Some(rng) = self.worker_rngs.get(index)
        {
            let mut guard = rng.lock().map_err(|_| HnswError::LockPoisoned {
                resource: "worker rng mutex",
            })?;
            return Ok(self.sample_level_from_rng(&mut guard));
        }

        let mut rng = self.rng.lock().map_err(|_| HnswError::LockPoisoned {
            resource: "rng mutex",
        })?;
        Ok(self.sample_level_from_rng(&mut rng))
    }

    /// Sample an HNSW level from a caller-provided random-number generator.
    pub(super) fn sample_level_from_rng(&self, rng: &mut SmallRng) -> usize {
        let mut level = 0_usize;
        while level < self.params.max_level() {
            let draw: f64 = rng.sample(Standard);
            if self.params.should_stop(draw) {
                break;
            }
            level += 1;
        }
        level
    }
}
