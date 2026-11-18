//! Parameter handling for the CPU HNSW implementation.
//!
//! Owns [`HnswParams`], the validated configuration that drives CPU index
//! construction: neighbour fan-out, construction search width, the sampled
//! level distribution and its cap, the insertion seed, and the distance-cache
//! settings. Two consumers share these values. The construction path uses the
//! sampler's stop rule and the effective search width, while
//! `crate::memory::estimate_peak_bytes_for_hnsw_params` uses
//! [`HnswParams::expected_layers_eighths`] to size the multi-layer edge
//! harvest. Keeping both in one type stops the memory estimate from drifting
//! away from the graph the builder actually produces.

use std::{num::NonZeroUsize, ops::Neg, time::Duration};

use num_traits::ToPrimitive;

use crate::hnsw::{distance_cache::DistanceCacheConfig, error::HnswError};

/// Greatest finite sampling draw strictly below one.
const LARGEST_DRAW_BELOW_ONE: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF);

/// Smallest draw the level sampler clamps a uniform sample to.
///
/// A continuation probability below this value can never be drawn, so the
/// sampler always stops and every node keeps only level zero.
const MIN_SAMPLED_DRAW: f64 = 1.0e-12;

/// Fixed-point scale that lets callers carry a fractional layer count.
///
/// The level expectation is a real number, while the memory estimator works in
/// integer arithmetic. Reporting eighths keeps a fractional layer meaningful
/// without floating point at the call site.
const LAYER_EIGHTHS: u64 = 8;

/// Configuration parameters for the CPU HNSW index.
#[derive(Clone, Debug, PartialEq)]
pub struct HnswParams {
    /// Maximum neighbours retained at the base graph layer.
    max_connections: usize,
    /// Construction-time search width.
    ef_construction: usize,
    /// Random-level distribution multiplier.
    level_multiplier: f64,
    /// Highest level a sampled node may receive.
    max_level: usize,
    /// Seed for deterministic insertion-level sampling.
    rng_seed: u64,
    /// Capacity and expiry configuration for distance reuse.
    distance_cache: DistanceCacheConfig,
}

/// Reasons parameter validation fails.
///
/// Shared by the production and Kani constructors so both map the same
/// checks to their own error representations.
#[derive(Clone, Copy, Debug)]
enum ParamsError {
    /// The configured neighbour fan-out is zero.
    ZeroMaxConnections,
    /// The construction search width is below the neighbour fan-out.
    EfBelowMaxConnections,
}

#[cfg(kani)]
impl ParamsError {
    /// Returns the static reason used by the Kani constructor.
    fn static_reason(self) -> &'static str {
        match self {
            Self::ZeroMaxConnections => "max_connections must be greater than zero",
            Self::EfBelowMaxConnections => "ef_construction must be at least max_connections",
        }
    }
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
        Self::validate_and_build(max_connections, ef_construction).map_err(|reason| match reason {
            ParamsError::ZeroMaxConnections => HnswError::InvalidParameters {
                reason: "max_connections must be greater than zero".into(),
            },
            ParamsError::EfBelowMaxConnections => HnswError::InvalidParameters {
                reason: format!(
                    "ef_construction ({ef_construction}) must be >= max_connections ({max_connections})"
                ),
            },
        })
    }

    /// Creates Kani parameters without constructing formatted production errors.
    #[cfg(kani)]
    pub(crate) fn new_for_kani(
        max_connections: usize,
        ef_construction: usize,
    ) -> Result<Self, &'static str> {
        Self::validate_and_build(max_connections, ef_construction)
            .map_err(ParamsError::static_reason)
    }

    /// Overrides the random level multiplier used when sampling layers.
    #[must_use]
    pub const fn with_level_multiplier(mut self, multiplier: f64) -> Self {
        self.level_multiplier = multiplier.max(f64::MIN_POSITIVE);
        self
    }

    /// Caps the maximum layer that will be sampled for new nodes.
    #[must_use]
    pub const fn with_max_level(mut self, max_level: usize) -> Self {
        self.max_level = max_level;
        self
    }

    /// Seeds the internal RNG to make insertion deterministic.
    #[must_use]
    pub const fn with_rng_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Applies a custom distance-cache configuration.
    #[must_use]
    pub const fn with_distance_cache_config(mut self, config: DistanceCacheConfig) -> Self {
        self.distance_cache = config;
        self
    }

    /// Overrides the maximum number of cached distances while preserving the
    /// existing cache time-to-live.
    #[must_use]
    pub const fn with_distance_cache_max_entries(mut self, max: NonZeroUsize) -> Self {
        self.distance_cache = self.distance_cache.with_max_entries(max);
        self
    }

    /// Overrides the optional time-to-live applied to cached entries.
    #[must_use]
    pub const fn with_distance_cache_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.distance_cache = self.distance_cache.with_ttl(ttl);
        self
    }

    /// Returns the neighbour fan-out enforced during insertion.
    #[must_use]
    pub const fn max_connections(&self) -> usize {
        self.max_connections
    }

    /// Returns the construction search breadth (`ef_construction`).
    #[must_use]
    pub const fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Return the highest permitted sampled graph level.
    pub(crate) const fn max_level(&self) -> usize {
        self.max_level
    }

    /// Return the expected number of graph layers a node occupies, in eighths.
    ///
    /// [`Self::should_stop`] continues while the clamped draw stays at or below
    /// `q = exp(-1 / level_multiplier)`, so a node reaches level `k` with
    /// probability `q^k`. Sampling stops at `max_level`, so the expectation is
    /// the finite sum `(1 - q^L) / (1 - q)` over `L = max_level + 1` layers.
    /// The infinite series `1 / (1 - q)` overstates it whenever the cap binds.
    /// The default multiplier of `1 / ln(M)` gives `q = 1 / M` and, for the
    /// levels that fit under the cap, the geometric mean `M / (M - 1)`.
    ///
    /// Two regimes need explicit handling. A multiplier large enough that `q`
    /// reaches the greatest permitted draw continues on every draw, so every
    /// node occupies the cap. A `q` below the sampler's minimum clamped draw
    /// can never be drawn, so every node keeps only level zero.
    ///
    /// The result is rounded up in eighths so callers can stay in integer
    /// arithmetic, and never falls below one layer: level zero is always
    /// occupied. At the default `M = 16` the count is nine eighths, matching
    /// `ceil(8 × M / (M - 1))`.
    pub(crate) fn expected_layers_eighths(&self) -> u64 {
        let layer_cap = u64::try_from(self.max_level)
            .unwrap_or(u64::MAX)
            .saturating_add(1)
            .saturating_mul(LAYER_EIGHTHS);
        // q = exp(-1 / multiplier); the multiplier is finite and positive.
        let sampled = self.level_multiplier.recip().neg().exp();
        if sampled >= LARGEST_DRAW_BELOW_ONE {
            // Every permitted draw continues, so the node reaches the cap.
            return layer_cap;
        }
        // The sampler clamps its draw, so a probability below the minimum
        // clamped draw can never be reached and every node keeps level zero.
        let continuation = if sampled < MIN_SAMPLED_DRAW {
            0.0
        } else {
            sampled
        };
        let levels = i32::try_from(self.max_level.saturating_add(1)).unwrap_or(i32::MAX);
        let cap_probability = continuation.powi(levels);
        // `(1 - q^L) / (1 - q)` with method calls rather than float operators,
        // which the restriction lints reject. `1 - q` is positive because the
        // cap-reaching regime returned above.
        let numerator = cap_probability.neg().mul_add(1.0, 1.0);
        let denominator = continuation.neg().mul_add(1.0, 1.0);
        let eighths = numerator
            .mul_add(denominator.recip(), 0.0)
            .mul_add(8.0, 0.0)
            .ceil();
        eighths
            .to_u64()
            .unwrap_or(u64::MAX)
            .clamp(LAYER_EIGHTHS, layer_cap)
    }

    /// Return the seed used by insertion-level sampling.
    pub(crate) const fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    /// Return the configuration used to build the distance cache.
    pub(crate) const fn distance_cache_config(&self) -> &DistanceCacheConfig {
        &self.distance_cache
    }

    /// Returns whether level sampling should terminate given a uniform draw.
    ///
    /// The multiplier of `1/ln(M)` induces a geometric tail where the chance of
    /// rising to the next layer is `1/M`, mirroring the reference algorithm.
    pub(crate) fn should_stop(&self, draw: f64) -> bool {
        let clamped = draw.clamp(MIN_SAMPLED_DRAW, LARGEST_DRAW_BELOW_ONE);
        clamped.ln().mul_add(self.level_multiplier.neg(), 0.0) < 1.0
    }

    /// Returns the construction search width useful for `point_count` items.
    ///
    /// The returned width preserves the `ef_construction >= max_connections`
    /// invariant while avoiding allocations for neighbours that cannot exist
    /// in the supplied batch.
    pub(crate) fn effective_ef_construction(&self, point_count: usize) -> usize {
        self.ef_construction
            .min(point_count.max(self.max_connections))
    }

    /// Caps construction search width to the useful width for a batch.
    pub(crate) fn bounded_for_point_count(mut self, point_count: usize) -> Self {
        self.ef_construction = self.effective_ef_construction(point_count);
        self
    }

    /// Validates the parameters and constructs the defaults.
    ///
    /// Shared by the production and Kani constructors; returns a static
    /// reason so the Kani path never constructs formatted errors, and keeps
    /// `max_level`, `rng_seed`, and `level_multiplier` defined once.
    fn validate_and_build(
        max_connections: usize,
        ef_construction: usize,
    ) -> Result<Self, ParamsError> {
        if max_connections == 0 {
            return Err(ParamsError::ZeroMaxConnections);
        }
        if ef_construction < max_connections {
            return Err(ParamsError::EfBelowMaxConnections);
        }
        Ok(Self {
            max_connections,
            ef_construction,
            level_multiplier: max_connections
                .to_f64()
                .unwrap_or(f64::INFINITY)
                .ln()
                .recip(),
            max_level: 12,
            rng_seed: 0x5EED_CAFE,
            distance_cache: DistanceCacheConfig::default(),
        })
    }
}

impl Default for HnswParams {
    /// Returns the documented default parameters of sixteen and sixty-four.
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 64,
            level_multiplier: (16.0_f64).ln().recip(),
            max_level: 12,
            rng_seed: 0x5EED_CAFE,
            distance_cache: DistanceCacheConfig::default(),
        }
    }
}

/// Returns the connection limit for a given level.
///
/// Level 0 (base layer) permits twice as many connections as higher levels,
/// following standard HNSW design to improve recall at the densest layer.
pub(crate) const fn connection_limit_for_level(level: usize, max_connections: usize) -> usize {
    if level == 0 {
        max_connections.saturating_mul(2)
    } else {
        max_connections
    }
}
