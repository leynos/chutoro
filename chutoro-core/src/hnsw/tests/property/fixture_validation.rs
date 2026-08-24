//! Property checks that validate generated HNSW vector fixtures.

use proptest::{prop_assert, prop_assert_eq, proptest};

use super::{
    strategies::hnsw_fixture_strategy,
    support::{dot, euclidean_distance, l2_norm},
    types::{DistributionMetadata, VectorDistribution},
};
use crate::DataSource;
use crate::test_utils::suite_proptest_config;

fn dimension_tolerance(radius: f32, dimension: usize) -> f32 {
    let scalar_dimension = f32::from(u16::try_from(dimension).unwrap_or(u16::MAX));
    radius.mul_add(scalar_dimension.sqrt(), 0.05)
}

proptest! {
    #![proptest_config(suite_proptest_config(256))]

    #[test]
    fn fixture_dimensions_are_consistent(fixture in hnsw_fixture_strategy()) {
        let dimension = fixture.dimension();
        prop_assert!(dimension > 0);
        prop_assert!(fixture.vectors.iter().all(|v| v.len() == dimension));
        prop_assert!(fixture.params.build().is_ok());
        let source = fixture.clone().into_source().expect("fixture must convert into a dense source");
        prop_assert_eq!(source.len(), fixture.vectors.len());
    }

    #[test]
    fn duplicate_groups_reference_identical_vectors(fixture in hnsw_fixture_strategy()) {
        if let DistributionMetadata::Duplicates { groups } = &fixture.metadata {
            for group in groups {
                let first = group.first().expect("duplicate group must contain at least one index");
                let Some(exemplar) = fixture.vectors.get(*first) else {
                    prop_assert!(false, "duplicate group index {first} is out of bounds");
                    continue;
                };
                for &index in group.iter().skip(1) {
                    let Some(vector) = fixture.vectors.get(index) else {
                        prop_assert!(false, "duplicate group index {index} is out of bounds");
                        continue;
                    };
                    prop_assert_eq!(vector, exemplar);
                }
            }
        }
    }

    #[test]
    fn distribution_matches_metadata(fixture in hnsw_fixture_strategy()) {
        match (&fixture.distribution, &fixture.metadata) {
            (VectorDistribution::Uniform, DistributionMetadata::Uniform { .. })
            | (VectorDistribution::Clustered, DistributionMetadata::Clustered { .. })
            | (VectorDistribution::Manifold, DistributionMetadata::Manifold { .. })
            | (VectorDistribution::Duplicates, DistributionMetadata::Duplicates { .. }) => {}
            (distribution, metadata) => prop_assert!(false, "distribution {distribution:?} mismatched metadata {metadata:?}"),
        }
    }

    #[test]
    fn cluster_points_remain_within_radius(fixture in hnsw_fixture_strategy()) {
        if let DistributionMetadata::Clustered { clusters } = &fixture.metadata {
            for cluster in clusters {
                let Some(end) = cluster.start.checked_add(cluster.len) else {
                    prop_assert!(false, "cluster range overflow");
                    continue;
                };
                let Some(points) = fixture.vectors.get(cluster.start..end) else {
                    prop_assert!(false, "cluster range is out of bounds");
                    continue;
                };
                for point in points {
                    let distance = euclidean_distance(point, &cluster.centroid);
                    let allowed = dimension_tolerance(cluster.radius, fixture.dimension());
                    prop_assert!(distance <= allowed, "point {point:?} exceeds radius: observed {distance}, allowed {allowed}");
                }
            }
        }
    }

    #[test]
    fn uniform_vectors_stay_within_bounds(fixture in hnsw_fixture_strategy()) {
        if let DistributionMetadata::Uniform { bound } = &fixture.metadata {
            let upper = bound.mul_add(1.0, f32::EPSILON);
            let lower = std::ops::Neg::neg(upper);
            for point in &fixture.vectors {
                for &value in point {
                    prop_assert!(value <= upper);
                    prop_assert!(value >= lower);
                }
            }
        }
    }

    #[test]
    fn manifold_points_stay_near_basis(fixture in hnsw_fixture_strategy()) {
        if let DistributionMetadata::Manifold { basis, noise_bound, origin, ambient_dim, intrinsic_dim } = &fixture.metadata {
            prop_assert_eq!(*ambient_dim, fixture.dimension());
            prop_assert_eq!(*intrinsic_dim, basis.len());
            prop_assert!(fixture.vectors.iter().all(|vector| vector.len() == *ambient_dim));
            prop_assert_eq!(origin.len(), *ambient_dim);
            prop_assert!(basis.iter().all(|basis_vector| basis_vector.len() == *ambient_dim));
            for point in &fixture.vectors {
                let mut diff: Vec<f32> = point
                    .iter()
                    .zip(origin)
                    .map(|(coordinate, origin_coordinate)| {
                        std::ops::Sub::sub(*coordinate, *origin_coordinate)
                    })
                    .collect();
                let mut projection = vec![0.0_f32; diff.len()];
                for basis_vec in basis {
                    let coeff = dot(&diff, basis_vec);
                    for (projection_component, component) in projection.iter_mut().zip(basis_vec) {
                        *projection_component = coeff.mul_add(*component, *projection_component);
                    }
                }
                for (value, projected_component) in diff.iter_mut().zip(&projection) {
                    *value = std::ops::Sub::sub(*value, *projected_component);
                }
                let residual = l2_norm(&diff);
                let tolerance = dimension_tolerance(*noise_bound, diff.len());
                prop_assert!(residual.total_cmp(&tolerance).is_le());
            }
        }
    }
}
