//! Synthetic data source for benchmarking.
//!
//! Provides [`SyntheticSource`], a [`DataSource`] implementation over
//! pre-generated N-dimensional f32 vectors with Euclidean distance.
//! Data is seeded for reproducibility across benchmark runs.

use chutoro_core::{DataSource, DataSourceError, MetricDescriptor};
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Errors that may occur during synthetic source generation.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum SyntheticError {
    /// The requested point count was zero.
    #[error("point count must be greater than zero")]
    ZeroPoints,
    /// The requested dimension count was zero.
    #[error("dimension count must be greater than zero")]
    ZeroDimensions,
}

/// Configuration for synthetic vector generation.
#[derive(Clone, Debug)]
pub struct SyntheticConfig {
    /// Number of points to generate.
    pub point_count: usize,
    /// Dimensionality of each vector.
    pub dimensions: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// A [`DataSource`] of random Euclidean vectors for benchmarking.
///
/// Vectors are stored in a flat row-major `Vec<f32>` and generated
/// eagerly from a seeded RNG. Distance is the standard Euclidean
/// metric (L2 norm).
///
/// # Examples
///
/// ```
/// use chutoro_benches::source::{SyntheticConfig, SyntheticSource};
/// use chutoro_core::DataSource;
///
/// let config = SyntheticConfig { point_count: 10, dimensions: 4, seed: 42 };
/// let source = SyntheticSource::generate(&config).expect("valid config");
/// assert_eq!(source.len(), 10);
/// ```
#[derive(Clone, Debug)]
pub struct SyntheticSource {
    data: Vec<f32>,
    point_count: usize,
    dimensions: usize,
}

impl SyntheticSource {
    /// Generates vectors eagerly from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SyntheticError::ZeroPoints`] if `point_count` is zero,
    /// or [`SyntheticError::ZeroDimensions`] if `dimensions` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use chutoro_benches::source::{SyntheticConfig, SyntheticSource};
    ///
    /// let config = SyntheticConfig { point_count: 5, dimensions: 3, seed: 7 };
    /// let source = SyntheticSource::generate(&config).expect("valid config");
    /// assert_eq!(source.len(), 5);
    /// ```
    pub fn generate(config: &SyntheticConfig) -> Result<Self, SyntheticError> {
        if config.point_count == 0 {
            return Err(SyntheticError::ZeroPoints);
        }
        if config.dimensions == 0 {
            return Err(SyntheticError::ZeroDimensions);
        }

        let total = config.point_count.saturating_mul(config.dimensions);
        let mut rng = SmallRng::seed_from_u64(config.seed);
        let data: Vec<f32> = (0..total)
            .map(|_| rng.gen_range(0.0_f32..1.0_f32))
            .collect();

        Ok(Self {
            data,
            point_count: config.point_count,
            dimensions: config.dimensions,
        })
    }

    /// Returns the dimensionality of each vector.
    #[must_use]
    pub const fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl DataSource for SyntheticSource {
    fn len(&self) -> usize {
        self.point_count
    }

    #[expect(
        clippy::unnecessary_literal_bound,
        reason = "DataSource trait constrains the return type to &str"
    )]
    fn name(&self) -> &str {
        "synthetic"
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("euclidean")
    }

    #[expect(
        clippy::float_arithmetic,
        reason = "Euclidean distance requires arithmetic on f32 values"
    )]
    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_start = left
            .checked_mul(self.dimensions)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let right_start = right
            .checked_mul(self.dimensions)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        let left_end = left_start
            .checked_add(self.dimensions)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let right_end = right_start
            .checked_add(self.dimensions)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;

        let left_vec = self
            .data
            .get(left_start..left_end)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let right_vec = self
            .data
            .get(right_start..right_end)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        let sum_sq = left_vec
            .iter()
            .zip(right_vec.iter())
            .fold(0.0_f32, |acc, (a, b)| {
                let diff = a - b;
                acc + diff * diff
            });
        Ok(sum_sq.sqrt())
    }
}

#[cfg(test)]
#[expect(
    clippy::float_arithmetic,
    reason = "distance comparison assertions require float arithmetic"
)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn default_config(point_count: usize, dimensions: usize) -> SyntheticConfig {
        SyntheticConfig {
            point_count,
            dimensions,
            seed: 42,
        }
    }

    // -- happy path: generation -------------------------------------------

    #[rstest]
    #[case::small(10, 4)]
    #[case::medium(100, 16)]
    #[case::large(500, 32)]
    fn generates_correct_point_count(#[case] point_count: usize, #[case] dimensions: usize) {
        let source = SyntheticSource::generate(&default_config(point_count, dimensions))
            .expect("generation must succeed");
        assert_eq!(source.len(), point_count);
        assert_eq!(source.dimensions(), dimensions);
    }

    // -- happy path: distance properties ----------------------------------

    #[rstest]
    fn distance_is_symmetric() {
        let source =
            SyntheticSource::generate(&default_config(20, 8)).expect("generation must succeed");
        let d_ij = source.distance(3, 7).expect("distance must succeed");
        let d_ji = source.distance(7, 3).expect("distance must succeed");
        assert!(
            (d_ij - d_ji).abs() < f32::EPSILON,
            "distance must be symmetric: d(3,7)={d_ij}, d(7,3)={d_ji}",
        );
    }

    #[rstest]
    fn distance_to_self_is_zero() {
        let source =
            SyntheticSource::generate(&default_config(10, 4)).expect("generation must succeed");
        for i in 0..source.len() {
            let d = source.distance(i, i).expect("distance must succeed");
            assert!(
                d.abs() < f32::EPSILON,
                "distance to self must be zero: d({i},{i})={d}",
            );
        }
    }

    #[rstest]
    fn distance_satisfies_triangle_inequality() {
        let source =
            SyntheticSource::generate(&default_config(20, 8)).expect("generation must succeed");
        let d_ab = source.distance(0, 1).expect("distance must succeed");
        let d_bc = source.distance(1, 2).expect("distance must succeed");
        let d_ac = source.distance(0, 2).expect("distance must succeed");
        assert!(
            d_ac <= d_ab + d_bc + f32::EPSILON,
            concat!(
                "triangle inequality violated: ",
                "d(0,2)={} > d(0,1)={} + d(1,2)={}",
            ),
            d_ac,
            d_ab,
            d_bc,
        );
    }

    #[rstest]
    fn distance_is_non_negative() {
        let source =
            SyntheticSource::generate(&default_config(10, 4)).expect("generation must succeed");
        for i in 0..source.len() {
            for j in 0..source.len() {
                let d = source.distance(i, j).expect("distance must succeed");
                assert!(d >= 0.0, "distance must be non-negative: d({i},{j})={d}");
            }
        }
    }

    // -- happy path: determinism ------------------------------------------

    #[rstest]
    fn deterministic_with_same_seed() {
        let config = default_config(50, 8);
        let a = SyntheticSource::generate(&config).expect("generation must succeed");
        let b = SyntheticSource::generate(&config).expect("generation must succeed");
        let d_a = a.distance(0, 1).expect("distance must succeed");
        let d_b = b.distance(0, 1).expect("distance must succeed");
        assert!(
            (d_a - d_b).abs() < f32::EPSILON,
            "same seed must produce identical distances",
        );
    }

    #[rstest]
    fn different_seeds_produce_different_data() {
        let a = SyntheticSource::generate(&SyntheticConfig {
            point_count: 50,
            dimensions: 8,
            seed: 1,
        })
        .expect("generation must succeed");
        let b = SyntheticSource::generate(&SyntheticConfig {
            point_count: 50,
            dimensions: 8,
            seed: 2,
        })
        .expect("generation must succeed");
        let d_a = a.distance(0, 1).expect("distance must succeed");
        let d_b = b.distance(0, 1).expect("distance must succeed");
        assert!(
            (d_a - d_b).abs() > f32::EPSILON,
            "different seeds should produce different distances",
        );
    }

    // -- unhappy path: generation errors ----------------------------------

    #[rstest]
    fn rejects_zero_points() {
        let err = SyntheticSource::generate(&default_config(0, 8))
            .expect_err("zero points must be rejected");
        assert_eq!(err, SyntheticError::ZeroPoints);
    }

    #[rstest]
    fn rejects_zero_dimensions() {
        let err = SyntheticSource::generate(&default_config(10, 0))
            .expect_err("zero dimensions must be rejected");
        assert_eq!(err, SyntheticError::ZeroDimensions);
    }

    // -- unhappy path: out-of-bounds index --------------------------------

    #[rstest]
    #[case(10, 0, 10, "left")]
    #[case(0, 10, 10, "right")]
    fn rejects_out_of_bounds_index(
        #[case] left: usize,
        #[case] right: usize,
        #[case] expected_index: usize,
        #[case] position: &str,
    ) {
        let source =
            SyntheticSource::generate(&default_config(5, 4)).expect("generation must succeed");
        let err = source
            .distance(left, right)
            .expect_err(&format!("out-of-bounds {position} index must be rejected"));
        assert!(
            matches!(err, DataSourceError::OutOfBounds { index } if index == expected_index),
            "expected OutOfBounds({expected_index}) for {position} index, got {err:?}",
        );
    }

    // -- edge case: name and metric descriptor ----------------------------

    #[rstest]
    fn source_name_is_synthetic() {
        let source =
            SyntheticSource::generate(&default_config(5, 4)).expect("generation must succeed");
        assert_eq!(source.name(), "synthetic");
    }

    #[rstest]
    fn source_is_not_empty() {
        let source =
            SyntheticSource::generate(&default_config(5, 4)).expect("generation must succeed");
        assert!(!source.is_empty());
    }
}
