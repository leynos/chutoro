//! Numeric synthetic data generators for benchmarking.

mod generation;

use crate::source::{SyntheticError, text::SyntheticTextConfig, text::SyntheticTextSource};
use chutoro_core::{DataSource, DataSourceError, MetricDescriptor};
use generation::{
    build_blob_centroids, manifold_point, resolve_axis_scales, validate_blob_config,
    validate_manifold_config,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};

/// Legacy uniform random vector configuration.
#[derive(Clone, Debug)]
pub struct SyntheticConfig {
    /// Number of points to generate.
    pub point_count: usize,
    /// Dimensionality of each vector.
    pub dimensions: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// Axis scaling strategy for Gaussian blob generation.
#[derive(Clone, Debug)]
pub enum Anisotropy {
    /// Uniform standard deviation scale for all dimensions.
    Isotropic(f32),
    /// Per-axis standard deviation scales.
    AxisScales(Vec<f32>),
}

/// Configuration for Gaussian blob synthetic data.
#[derive(Clone, Debug)]
pub struct GaussianBlobConfig {
    /// Number of points to generate.
    pub point_count: usize,
    /// Dimensionality of each vector.
    pub dimensions: usize,
    /// Number of Gaussian clusters.
    pub cluster_count: usize,
    /// Minimum centroid separation scale.
    pub separation: f32,
    /// Covariance anisotropy control.
    pub anisotropy: Anisotropy,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// Supported non-linear manifold patterns.
#[derive(Clone, Copy, Debug)]
pub enum ManifoldPattern {
    /// A noisy ring embedded in `dimensions >= 2`.
    Ring,
    /// A Swiss-roll manifold embedded in `dimensions >= 3`.
    SwissRoll,
}

/// Configuration for non-linear manifold generators.
#[derive(Clone, Debug)]
pub struct ManifoldConfig {
    /// Number of points to generate.
    pub point_count: usize,
    /// Dimensionality of each vector.
    pub dimensions: usize,
    /// Manifold pattern family.
    pub pattern: ManifoldPattern,
    /// Primary radius scale.
    pub major_radius: f32,
    /// Thickness/noise radius around the manifold.
    pub thickness: f32,
    /// Number of turns for Swiss-roll generation.
    pub turns: usize,
    /// Additive Gaussian noise multiplier.
    pub noise: f32,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// A numeric synthetic [`DataSource`] using Euclidean distance.
#[derive(Clone, Debug)]
pub struct SyntheticSource {
    data: Vec<f32>,
    point_count: usize,
    dimensions: usize,
    name: &'static str,
}

impl SyntheticSource {
    /// Generates uniform random vectors in `[0.0, 1.0)`.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] when the configuration is invalid.
    pub fn generate(config: &SyntheticConfig) -> Result<Self, SyntheticError> {
        validate_basic_numeric_config(config.point_count, config.dimensions)?;
        let total = checked_total(config.point_count, config.dimensions)?;

        let mut rng = SmallRng::seed_from_u64(config.seed);
        let data: Vec<f32> = (0..total)
            .map(|_| rng.gen_range(0.0_f32..1.0_f32))
            .collect();

        Self::from_parts(
            "synthetic-uniform",
            data,
            config.point_count,
            config.dimensions,
        )
    }

    /// Generates Gaussian blobs with configurable separation and anisotropy.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] when the configuration is invalid.
    pub fn generate_gaussian_blobs(config: &GaussianBlobConfig) -> Result<Self, SyntheticError> {
        let (data, _labels) = Self::generate_gaussian_blob_data(config)?;

        Self::from_parts(
            "synthetic-gaussian-blobs",
            data,
            config.point_count,
            config.dimensions,
        )
    }

    /// Generates Gaussian blobs and returns deterministic ground-truth labels.
    ///
    /// Labels are assigned in round-robin centroid order so each generated
    /// point can be compared against benchmark cluster assignments.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] when the configuration is invalid.
    pub fn generate_gaussian_blobs_with_labels(
        config: &GaussianBlobConfig,
    ) -> Result<(Self, Vec<usize>), SyntheticError> {
        let (data, labels) = Self::generate_gaussian_blob_data(config)?;
        let source = Self::from_parts(
            "synthetic-gaussian-blobs",
            data,
            config.point_count,
            config.dimensions,
        )?;
        Ok((source, labels))
    }

    #[expect(
        clippy::float_arithmetic,
        reason = "Gaussian data generation requires floating-point arithmetic"
    )]
    fn generate_gaussian_blob_data(
        config: &GaussianBlobConfig,
    ) -> Result<(Vec<f32>, Vec<usize>), SyntheticError> {
        validate_basic_numeric_config(config.point_count, config.dimensions)?;
        validate_blob_config(config)?;

        let scales = resolve_axis_scales(&config.anisotropy, config.dimensions)?;
        let centroids = build_blob_centroids(config, &mut SmallRng::seed_from_u64(config.seed));
        let mut rng = SmallRng::seed_from_u64(config.seed ^ 0xA5A5_A5A5_A5A5_A5A5_u64);
        let total = checked_total(config.point_count, config.dimensions)?;
        let mut data = Vec::with_capacity(total);
        let mut labels = Vec::with_capacity(config.point_count);
        let mut current_label = 0usize;
        for centroid in centroids.iter().cycle().take(config.point_count) {
            labels.push(current_label);
            current_label = current_label.saturating_add(1);
            if current_label == config.cluster_count {
                current_label = 0;
            }
            for (centroid_value, scale) in centroid.iter().zip(&scales) {
                let sample = generation::standard_normal_sample(&mut rng)?;
                data.push(*centroid_value + sample * *scale);
            }
        }

        Ok((data, labels))
    }

    /// Generates a non-linearly-separable manifold pattern.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] when the configuration is invalid.
    pub fn generate_manifold(config: &ManifoldConfig) -> Result<Self, SyntheticError> {
        validate_basic_numeric_config(config.point_count, config.dimensions)?;
        validate_manifold_config(config)?;

        let total = checked_total(config.point_count, config.dimensions)?;
        let mut data = Vec::with_capacity(total);
        let mut rng = SmallRng::seed_from_u64(config.seed);

        for _ in 0..config.point_count {
            let point = manifold_point(config, &mut rng)?;
            data.extend(point);
        }

        let name = match config.pattern {
            ManifoldPattern::Ring => "synthetic-manifold-ring",
            ManifoldPattern::SwissRoll => "synthetic-manifold-swiss-roll",
        };

        Self::from_parts(name, data, config.point_count, config.dimensions)
    }

    /// Creates a synthetic text source that uses Levenshtein distance.
    ///
    /// # Errors
    /// Returns [`SyntheticError`] when the text configuration is invalid.
    pub fn generate_text(
        config: &SyntheticTextConfig,
    ) -> Result<SyntheticTextSource, SyntheticError> {
        SyntheticTextSource::generate(config)
    }

    /// Returns the dimensionality of each vector.
    #[must_use]
    #[rustfmt::skip]
    pub const fn dimensions(&self) -> usize { self.dimensions }

    pub(crate) fn from_parts(
        name: &'static str,
        data: Vec<f32>,
        point_count: usize,
        dimensions: usize,
    ) -> Result<Self, SyntheticError> {
        validate_basic_numeric_config(point_count, dimensions)?;
        let expected = checked_total(point_count, dimensions)?;
        if data.len() != expected {
            return Err(SyntheticError::InvalidFloatParameter {
                parameter: "data.len()",
            });
        }

        Ok(Self {
            data,
            point_count,
            dimensions,
            name,
        })
    }
}

impl DataSource for SyntheticSource {
    #[rustfmt::skip]
    fn len(&self) -> usize { self.point_count }

    #[rustfmt::skip]
    fn name(&self) -> &str { self.name }

    #[rustfmt::skip]
    fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("euclidean") }

    #[expect(
        clippy::float_arithmetic,
        reason = "Euclidean distance requires floating-point arithmetic"
    )]
    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_start = checked_offset(left, self.dimensions)?;
        let right_start = checked_offset(right, self.dimensions)?;
        let left_end = checked_end(left_start, self.dimensions, left)?;
        let right_end = checked_end(right_start, self.dimensions, right)?;

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
            .zip(right_vec)
            .fold(0.0_f32, |acc, (lhs, rhs)| {
                let diff = lhs - rhs;
                acc + diff * diff
            });
        Ok(sum_sq.sqrt())
    }
}

const fn validate_basic_numeric_config(
    point_count: usize,
    dimensions: usize,
) -> Result<(), SyntheticError> {
    if point_count == 0 {
        return Err(SyntheticError::ZeroPoints);
    }
    if dimensions == 0 {
        return Err(SyntheticError::ZeroDimensions);
    }
    Ok(())
}

fn checked_total(point_count: usize, dimensions: usize) -> Result<usize, SyntheticError> {
    point_count
        .checked_mul(dimensions)
        .ok_or(SyntheticError::Overflow)
}

fn checked_offset(index: usize, dimensions: usize) -> Result<usize, DataSourceError> {
    index
        .checked_mul(dimensions)
        .ok_or(DataSourceError::OutOfBounds { index })
}

fn checked_end(start: usize, dimensions: usize, index: usize) -> Result<usize, DataSourceError> {
    start
        .checked_add(dimensions)
        .ok_or(DataSourceError::OutOfBounds { index })
}
