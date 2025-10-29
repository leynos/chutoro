//! Parameter handling for the CPU HNSW implementation.

use std::{num::NonZeroUsize, time::Duration};

use crate::hnsw::{distance_cache::DistanceCacheConfig, error::HnswError};

/// Configuration parameters for the CPU HNSW index.
#[derive(Clone, Debug)]
pub struct HnswParams {
    max_connections: usize,
    ef_construction: usize,
    level_multiplier: f64,
    max_level: usize,
    rng_seed: u64,
    distance_cache: DistanceCacheConfig,
}

impl HnswParams {
    /// Creates a new parameter set with explicit neighbour and search widths.
    ///
    /// # Errors
    /// Returns [`HnswError::InvalidParameters`] when `max_connections` is zero or
    /// when `ef_construction` is smaller than `max_connections`.
    ///
    /// # Examples
    /// ```
    /// use chutoro_core::HnswParams;
    /// let params = HnswParams::new(16, 64).expect("parameters must be valid");
    /// assert_eq!(params.max_connections(), 16);
    /// ```
    pub fn new(max_connections: usize, ef_construction: usize) -> Result<Self, HnswError> {
        if max_connections == 0 {
            return Err(HnswError::InvalidParameters {
                reason: "max_connections must be greater than zero".into(),
            });
        }
        if ef_construction < max_connections {
            return Err(HnswError::InvalidParameters {
                reason: format!(
                    "ef_construction ({ef_construction}) must be >= max_connections ({max_connections})"
                ),
            });
        }
        Ok(Self {
            max_connections,
            ef_construction,
            level_multiplier: (max_connections as f64).ln().recip(),
            max_level: 12,
            rng_seed: 0x5EED_CAFE,
            distance_cache: DistanceCacheConfig::default(),
        })
    }

    /// Overrides the random level multiplier used when sampling layers.
    #[must_use]
    pub fn with_level_multiplier(mut self, multiplier: f64) -> Self {
        self.level_multiplier = multiplier.max(f64::MIN_POSITIVE);
        self
    }

    /// Caps the maximum layer that will be sampled for new nodes.
    #[must_use]
    pub fn with_max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
    }

    /// Seeds the internal RNG to make insertion deterministic.
    #[must_use]
    pub fn with_rng_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Applies a custom distance-cache configuration.
    #[must_use]
    pub fn with_distance_cache_config(mut self, config: DistanceCacheConfig) -> Self {
        self.distance_cache = config;
        self
    }

    /// Overrides the maximum number of cached distances.
    #[must_use]
    pub fn with_distance_cache_max_entries(mut self, max: NonZeroUsize) -> Self {
        self.distance_cache = DistanceCacheConfig::new(max).with_ttl(self.distance_cache.ttl());
        self
    }

    /// Overrides the optional time-to-live applied to cached entries.
    #[must_use]
    pub fn with_distance_cache_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.distance_cache = self.distance_cache.clone().with_ttl(ttl);
        self
    }

    /// Returns the neighbour fan-out enforced during insertion.
    #[must_use]
    pub fn max_connections(&self) -> usize {
        self.max_connections
    }

    /// Returns the construction search breadth (`ef_construction`).
    #[must_use]
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    pub(crate) fn max_level(&self) -> usize {
        self.max_level
    }

    pub(crate) fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    pub(crate) fn distance_cache_config(&self) -> &DistanceCacheConfig {
        &self.distance_cache
    }

    /// Returns whether level sampling should terminate given a uniform draw.
    ///
    /// The multiplier of `1/ln(M)` induces a geometric tail where the chance of
    /// rising to the next layer is `1/M`, mirroring the reference algorithm.
    pub(crate) fn should_stop(&self, draw: f64) -> bool {
        let clamped = draw.clamp(1.0e-12, 1.0 - f64::EPSILON);
        (-clamped.ln()) * self.level_multiplier < 1.0
    }
}

impl Default for HnswParams {
    fn default() -> Self {
        Self::new(16, 64).expect("default parameters must be valid")
    }
}
