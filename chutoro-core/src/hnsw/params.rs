//! Parameter handling for the CPU HNSW implementation.

use crate::hnsw::error::HnswError;

/// Configuration parameters for the CPU HNSW index.
#[derive(Clone, Debug)]
pub struct HnswParams {
    max_connections: usize,
    ef_construction: usize,
    level_multiplier: f64,
    max_level: usize,
    rng_seed: u64,
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
            level_multiplier: (max_connections as f64).ln().recip().max(1.0),
            max_level: 12,
            rng_seed: 0x5EED_CAFE,
        })
    }

    /// Overrides the random level multiplier used when sampling layers.
    #[must_use]
    pub fn with_level_multiplier(mut self, multiplier: f64) -> Self {
        self.level_multiplier = multiplier.max(1.0);
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
