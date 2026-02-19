//! Numeric generator helper routines.

use super::{Anisotropy, GaussianBlobConfig, ManifoldConfig, ManifoldPattern};
use crate::source::SyntheticError;
use rand::{Rng, rngs::SmallRng};
use std::f32::consts::PI;

pub(super) fn validate_blob_config(config: &GaussianBlobConfig) -> Result<(), SyntheticError> {
    if config.cluster_count == 0 {
        return Err(SyntheticError::ZeroClusters);
    }
    if config.cluster_count > config.point_count {
        return Err(SyntheticError::ClusterCountExceedsPointCount {
            cluster_count: config.cluster_count,
            point_count: config.point_count,
        });
    }
    if !config.separation.is_finite() || config.separation <= 0.0 {
        return Err(SyntheticError::InvalidFloatParameter {
            parameter: "separation",
        });
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "centroid placement uses index-derived floating-point angles"
)]
#[expect(
    clippy::float_arithmetic,
    reason = "centroid placement uses trigonometric expressions"
)]
pub(super) fn build_blob_centroids(
    config: &GaussianBlobConfig,
    rng: &mut SmallRng,
) -> Vec<Vec<f32>> {
    (0..config.cluster_count)
        .map(|cluster_index| {
            let angle = (cluster_index as f32 / config.cluster_count as f32) * (2.0 * PI);
            let mut centroid = vec![0.0_f32; config.dimensions];
            if let Some(value) = centroid.get_mut(0) {
                *value = config.separation * angle.cos();
            }
            if let Some(value) = centroid.get_mut(1) {
                *value = config.separation * angle.sin();
            }
            for value in centroid.iter_mut().skip(2) {
                *value = rng.gen_range((-0.2 * config.separation)..(0.2 * config.separation));
            }
            centroid
        })
        .collect()
}

pub(super) fn resolve_axis_scales(
    anisotropy: &Anisotropy,
    dimensions: usize,
) -> Result<Vec<f32>, SyntheticError> {
    match anisotropy {
        Anisotropy::Isotropic(scale) => validate_isotropic_scale(*scale, dimensions),
        Anisotropy::AxisScales(scales) => validate_axis_scales(scales, dimensions),
    }
}

fn validate_isotropic_scale(scale: f32, dimensions: usize) -> Result<Vec<f32>, SyntheticError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SyntheticError::InvalidFloatParameter {
            parameter: "anisotropy.scale",
        });
    }
    Ok(vec![scale; dimensions])
}

fn validate_axis_scales(scales: &[f32], dimensions: usize) -> Result<Vec<f32>, SyntheticError> {
    if scales.len() != dimensions {
        return Err(SyntheticError::AxisScaleLengthMismatch {
            expected: dimensions,
            actual: scales.len(),
        });
    }
    for (index, value) in scales.iter().enumerate() {
        if !value.is_finite() || *value <= 0.0 {
            return Err(SyntheticError::InvalidAxisScale { index });
        }
    }
    Ok(scales.to_vec())
}

fn validate_float_param(
    value: f32,
    parameter: &'static str,
    allow_zero: bool,
) -> Result<(), SyntheticError> {
    let is_valid = if allow_zero {
        value.is_finite() && value >= 0.0
    } else {
        value.is_finite() && value > 0.0
    };

    if is_valid {
        Ok(())
    } else {
        Err(SyntheticError::InvalidFloatParameter { parameter })
    }
}

const fn validate_pattern_dimensions(
    pattern: &'static str,
    minimum: usize,
    actual: usize,
) -> Result<(), SyntheticError> {
    if actual < minimum {
        return Err(SyntheticError::InsufficientManifoldDimensions {
            pattern,
            minimum,
            actual,
        });
    }
    Ok(())
}

const fn validate_ring_pattern(config: &ManifoldConfig) -> Result<(), SyntheticError> {
    validate_pattern_dimensions("ring", 2, config.dimensions)
}

const fn validate_swiss_roll_pattern(config: &ManifoldConfig) -> Result<(), SyntheticError> {
    if config.dimensions < 3 {
        return validate_pattern_dimensions("swiss_roll", 3, config.dimensions);
    }
    if config.turns == 0 {
        return Err(SyntheticError::ZeroTurns);
    }
    Ok(())
}

pub(super) fn validate_manifold_config(config: &ManifoldConfig) -> Result<(), SyntheticError> {
    validate_float_param(config.major_radius, "major_radius", false)?;
    validate_float_param(config.thickness, "thickness", true)?;
    validate_float_param(config.noise, "noise", true)?;

    match config.pattern {
        ManifoldPattern::Ring => validate_ring_pattern(config),
        ManifoldPattern::SwissRoll => validate_swiss_roll_pattern(config),
    }
}

#[expect(
    clippy::float_arithmetic,
    reason = "sampling manifold coordinates requires floating-point arithmetic"
)]
fn generate_ring_point(
    point: &mut [f32],
    config: &ManifoldConfig,
    rng: &mut SmallRng,
) -> Result<(), SyntheticError> {
    let theta = rng.gen_range(0.0_f32..(2.0 * PI));
    let radial_noise = standard_normal_sample(rng)? * config.thickness;
    let radius = config.major_radius + radial_noise;
    if let Some(value) = point.get_mut(0) {
        *value = radius * theta.cos();
    }
    if let Some(value) = point.get_mut(1) {
        *value = radius * theta.sin();
    }
    for value in point.iter_mut().skip(2) {
        *value = standard_normal_sample(rng)? * config.noise;
    }
    Ok(())
}

#[expect(
    clippy::float_arithmetic,
    reason = "sampling manifold coordinates requires floating-point arithmetic"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "turn count is converted to f32 for angle calculations"
)]
fn generate_swiss_roll_point(
    point: &mut [f32],
    config: &ManifoldConfig,
    rng: &mut SmallRng,
) -> Result<(), SyntheticError> {
    let max_t = config.turns as f32 * 2.0 * PI;
    let t = rng.gen_range(0.0_f32..max_t);
    let radial_noise = standard_normal_sample(rng)? * config.thickness;
    let radial = config.major_radius + t + radial_noise;
    let x_noise = standard_normal_sample(rng)? * config.noise;
    let y_noise = standard_normal_sample(rng)? * config.noise;
    let z_noise = standard_normal_sample(rng)? * config.noise;
    if let Some(value) = point.get_mut(0) {
        *value = radial * t.cos() + x_noise;
    }
    if let Some(value) = point.get_mut(1) {
        *value = rng.gen_range(-config.major_radius..config.major_radius) + y_noise;
    }
    if let Some(value) = point.get_mut(2) {
        *value = radial * t.sin() + z_noise;
    }
    for value in point.iter_mut().skip(3) {
        *value = standard_normal_sample(rng)? * config.noise;
    }
    Ok(())
}

pub(super) fn manifold_point(
    config: &ManifoldConfig,
    rng: &mut SmallRng,
) -> Result<Vec<f32>, SyntheticError> {
    let mut point = vec![0.0_f32; config.dimensions];
    match config.pattern {
        ManifoldPattern::Ring => generate_ring_point(&mut point, config, rng)?,
        ManifoldPattern::SwissRoll => generate_swiss_roll_point(&mut point, config, rng)?,
    }

    Ok(point)
}

#[expect(
    clippy::float_arithmetic,
    reason = "Box-Muller transform requires floating-point arithmetic"
)]
pub(super) fn standard_normal_sample(rng: &mut SmallRng) -> Result<f32, SyntheticError> {
    let mut u1 = rng.gen_range(0.0_f32..1.0_f32);
    if u1 <= f32::EPSILON {
        u1 = f32::EPSILON;
    }
    let u2 = rng.gen_range(0.0_f32..1.0_f32);
    let radius = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * PI * u2;
    let sample = radius * theta.cos();
    if sample.is_finite() {
        Ok(sample)
    } else {
        Err(SyntheticError::InvalidFloatParameter {
            parameter: "standard_normal_sample",
        })
    }
}
