//! Unit tests for synthetic source generators.

use super::{
    Anisotropy, GaussianBlobConfig, ManifoldConfig, ManifoldPattern, SyntheticConfig,
    SyntheticError, SyntheticSource, SyntheticTextConfig,
};
use chutoro_core::DataSource;
use rstest::{fixture, rstest};

#[fixture]
fn uniform_config() -> SyntheticConfig {
    SyntheticConfig {
        point_count: 32,
        dimensions: 8,
        seed: 7,
    }
}

#[fixture]
fn gaussian_config() -> GaussianBlobConfig {
    GaussianBlobConfig {
        point_count: 120,
        dimensions: 8,
        cluster_count: 3,
        separation: 4.0,
        anisotropy: Anisotropy::Isotropic(0.4),
        seed: 11,
    }
}

#[fixture]
fn ring_config() -> ManifoldConfig {
    ManifoldConfig {
        point_count: 64,
        dimensions: 4,
        pattern: ManifoldPattern::Ring,
        major_radius: 5.0,
        thickness: 0.5,
        turns: 1,
        noise: 0.2,
        seed: 19,
    }
}

#[fixture]
fn text_config() -> SyntheticTextConfig {
    SyntheticTextConfig {
        item_count: 48,
        min_length: 4,
        max_length: 10,
        seed: 13,
        alphabet: "abcdefg".to_owned(),
        template_words: vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()],
        max_edits_per_item: 3,
    }
}

#[rstest]
#[case::small(8, 2)]
#[case::medium(128, 16)]
#[case::larger(300, 32)]
fn uniform_generator_respects_shape(#[case] point_count: usize, #[case] dimensions: usize) {
    let source = SyntheticSource::generate(&SyntheticConfig {
        point_count,
        dimensions,
        seed: 5,
    })
    .expect("uniform generation should succeed");

    assert_eq!(source.len(), point_count);
    assert_eq!(source.dimensions(), dimensions);
}

#[rstest]
fn uniform_generator_rejects_zero_points(uniform_config: SyntheticConfig) {
    let error = SyntheticSource::generate(&SyntheticConfig {
        point_count: 0,
        ..uniform_config
    })
    .expect_err("zero points must fail");
    assert!(matches!(error, SyntheticError::ZeroPoints));
}

#[rstest]
fn gaussian_generator_rejects_cluster_overflow(gaussian_config: GaussianBlobConfig) {
    let error = SyntheticSource::generate_gaussian_blobs(&GaussianBlobConfig {
        cluster_count: gaussian_config.point_count + 1,
        ..gaussian_config
    })
    .expect_err("cluster count larger than point count must fail");

    assert!(matches!(
        error,
        SyntheticError::ClusterCountExceedsPointCount { .. }
    ));
}

#[rstest]
fn gaussian_generator_accepts_axis_anisotropy(gaussian_config: GaussianBlobConfig) {
    let source = SyntheticSource::generate_gaussian_blobs(&GaussianBlobConfig {
        anisotropy: Anisotropy::AxisScales(vec![1.0, 0.8, 0.5, 1.2, 0.9, 1.1, 0.7, 0.6]),
        ..gaussian_config
    })
    .expect("axis anisotropy should be accepted");

    assert_eq!(source.len(), 120);
    assert_eq!(source.dimensions(), 8);
}

#[rstest]
#[expect(
    clippy::float_arithmetic,
    reason = "test compares generated distances via subtraction"
)]
fn gaussian_generator_is_deterministic(gaussian_config: GaussianBlobConfig) {
    let left = SyntheticSource::generate_gaussian_blobs(&gaussian_config)
        .expect("first gaussian generation should succeed");
    let right = SyntheticSource::generate_gaussian_blobs(&gaussian_config)
        .expect("second gaussian generation should succeed");

    let left_distance = left
        .distance(1, 7)
        .expect("distance lookup should succeed for left source");
    let right_distance = right
        .distance(1, 7)
        .expect("distance lookup should succeed for right source");

    assert!((left_distance - right_distance).abs() < f32::EPSILON);
}

#[rstest]
fn manifold_ring_requires_at_least_two_dimensions(ring_config: ManifoldConfig) {
    let error = SyntheticSource::generate_manifold(&ManifoldConfig {
        dimensions: 1,
        ..ring_config
    })
    .expect_err("ring manifold with one dimension must fail");

    assert!(matches!(
        error,
        SyntheticError::InsufficientManifoldDimensions {
            pattern: "ring",
            ..
        }
    ));
}

#[rstest]
fn manifold_swiss_roll_requires_turns(ring_config: ManifoldConfig) {
    let error = SyntheticSource::generate_manifold(&ManifoldConfig {
        pattern: ManifoldPattern::SwissRoll,
        dimensions: 3,
        turns: 0,
        ..ring_config
    })
    .expect_err("Swiss-roll without turns must fail");

    assert!(matches!(error, SyntheticError::ZeroTurns));
}

#[rstest]
fn manifold_generator_produces_expected_shape(ring_config: ManifoldConfig) {
    let source = SyntheticSource::generate_manifold(&ring_config)
        .expect("ring manifold generation should succeed");

    assert_eq!(source.len(), ring_config.point_count);
    assert_eq!(source.dimensions(), ring_config.dimensions);
}

#[rstest]
fn text_generator_respects_bounds(text_config: SyntheticTextConfig) {
    let source = SyntheticSource::generate_text(&text_config)
        .expect("text generator should succeed for valid config");

    assert_eq!(source.len(), text_config.item_count);
    for entry in source.lines() {
        let len = entry.chars().count();
        assert!(len >= text_config.min_length);
        assert!(len <= text_config.max_length);
    }
}

#[rstest]
fn text_generator_rejects_invalid_lengths(text_config: SyntheticTextConfig) {
    let error = SyntheticSource::generate_text(&SyntheticTextConfig {
        min_length: 12,
        max_length: 8,
        ..text_config
    })
    .expect_err("inverted length range must fail");

    assert!(matches!(
        error,
        SyntheticError::InvalidTextLengthRange { .. }
    ));
}

#[rstest]
#[expect(
    clippy::float_arithmetic,
    reason = "test compares Levenshtein distances via subtraction"
)]
fn text_distance_is_symmetric(text_config: SyntheticTextConfig) {
    let source =
        SyntheticSource::generate_text(&text_config).expect("text generation should succeed");

    let left_right = source
        .distance(0, 1)
        .expect("left-right distance should succeed");
    let right_left = source
        .distance(1, 0)
        .expect("right-left distance should succeed");

    assert!((left_right - right_left).abs() < f32::EPSILON);
}
