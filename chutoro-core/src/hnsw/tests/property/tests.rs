use proptest::{
    prelude::any,
    prop_assert, prop_assert_eq, proptest,
    test_runner::{TestCaseError, TestCaseResult, TestError, TestRunner},
};
use rstest::rstest;

use super::{
    search_property::run_search_correctness_property,
    strategies::hnsw_fixture_strategy,
    support::{DenseVectorSource, dot, euclidean_distance, l2_norm},
    types::{DistributionMetadata, HnswParamsSeed, VectorDistribution},
};
use crate::DataSource;
use crate::error::DataSourceError;
use crate::hnsw::HnswError;

#[test]
fn dense_vector_source_rejects_inconsistent_rows() {
    let err = DenseVectorSource::new("empty", Vec::new()).expect_err("empty data should fail");
    assert_eq!(err, DataSourceError::EmptyData);

    let err = DenseVectorSource::new("zero", vec![vec![]]).expect_err("zero dimension should fail");
    assert_eq!(err, DataSourceError::ZeroDimension);

    let err = DenseVectorSource::new("mismatch", vec![vec![0.0, 1.0], vec![1.0]])
        .expect_err("dimension mismatch must fail");
    assert_eq!(
        err,
        DataSourceError::DimensionMismatch { left: 2, right: 1 },
    );
}

#[rstest]
#[case(0, 0)]
#[case(4, 3)]
fn params_seed_build_propagates_errors(
    #[case] max_connections: usize,
    #[case] ef_construction: usize,
) {
    let seed = HnswParamsSeed {
        max_connections,
        ef_construction,
        level_multiplier: 0.5,
        max_level: 4,
        rng_seed: 7,
    };
    let err = seed.build().expect_err("invalid parameters must fail");
    assert!(matches!(err, HnswError::InvalidParameters { .. }));
}

proptest! {
    #[test]
    fn fixture_dimensions_are_consistent(fixture in hnsw_fixture_strategy()) {
        let dimension = fixture.dimension();
        prop_assert!(dimension > 0);
        prop_assert!(fixture.vectors.iter().all(|v| v.len() == dimension));
        prop_assert!(fixture.params.build().is_ok());
        let source = fixture
            .clone()
            .into_source()
            .expect("fixture must convert into a dense source");
        prop_assert_eq!(source.len(), fixture.vectors.len());
    }

    #[test]
    fn duplicate_groups_reference_identical_vectors(
        fixture in hnsw_fixture_strategy()
    ) {
        if let DistributionMetadata::Duplicates { groups } = &fixture.metadata {
            for group in groups {
                let first = group.first().expect("duplicate group must contain at least one index");
                prop_assert!(*first < fixture.vectors.len());
                let exemplar = &fixture.vectors[*first];
                for &index in group.iter().skip(1) {
                    prop_assert!(index < fixture.vectors.len());
                    prop_assert!(fixture.vectors[index] == *exemplar);
                }
            }
        }
    }

    #[test]
    fn distribution_matches_metadata(fixture in hnsw_fixture_strategy()) {
        match (&fixture.distribution, &fixture.metadata) {
            (VectorDistribution::Uniform, DistributionMetadata::Uniform { .. }) => {}
            (VectorDistribution::Clustered, DistributionMetadata::Clustered { .. }) => {}
            (VectorDistribution::Manifold, DistributionMetadata::Manifold { .. }) => {}
            (VectorDistribution::Duplicates, DistributionMetadata::Duplicates { .. }) => {}
            (distribution, metadata) => {
                prop_assert!(
                    false,
                    "distribution {:?} mismatched metadata {:?}",
                    distribution,
                    metadata,
                );
            }
        }
    }

    #[test]
    fn cluster_points_remain_within_radius(
        fixture in hnsw_fixture_strategy()
    ) {
        if let DistributionMetadata::Clustered { clusters } = &fixture.metadata {
            for cluster in clusters {
                for point in &fixture.vectors[cluster.start..cluster.start + cluster.len] {
                    let distance = euclidean_distance(point, &cluster.centroid);
                    prop_assert!(
                        distance <= cluster.radius * (fixture.dimension() as f32).sqrt() + 0.05,
                        "point {:?} exceeds radius: observed {}, allowed {}",
                        point,
                        distance,
                        cluster.radius * (fixture.dimension() as f32).sqrt() + 0.05,
                    );
                }
            }
        }
    }

    #[test]
    fn uniform_vectors_stay_within_bounds(
        fixture in hnsw_fixture_strategy()
    ) {
        if let DistributionMetadata::Uniform { bound } = &fixture.metadata {
            for point in &fixture.vectors {
                for &value in point {
                    prop_assert!(value <= bound + f32::EPSILON);
                    prop_assert!(value >= -bound - f32::EPSILON);
                }
            }
        }
    }

    #[test]
    fn manifold_points_stay_near_basis(
        fixture in hnsw_fixture_strategy()
    ) {
        if let DistributionMetadata::Manifold {
            basis,
            noise_bound,
            origin,
            ambient_dim,
            intrinsic_dim,
        } = &fixture.metadata
        {
            prop_assert_eq!(*ambient_dim, fixture.dimension());
            prop_assert_eq!(*intrinsic_dim, basis.len());
            for point in &fixture.vectors {
                let mut diff: Vec<f32> = point.iter().zip(origin.iter()).map(|(p, o)| p - o).collect();
                let mut projection = vec![0.0_f32; diff.len()];
                for basis_vec in basis.iter() {
                    let coeff = dot(&diff, basis_vec);
                    for (proj, component) in projection.iter_mut().zip(basis_vec) {
                        *proj += coeff * component;
                    }
                }
                for (value, proj) in diff.iter_mut().zip(&projection) {
                    *value -= proj;
                }
                let residual = l2_norm(&diff);
                let tolerance = (*noise_bound * diff.len() as f32).sqrt() + 0.05;
                prop_assert!(residual <= tolerance);
            }
        }
    }
}

#[test]
fn hnsw_search_matches_brute_force_proptest() -> TestCaseResult {
    let mut runner = TestRunner::default();
    runner
        .run(
            &(hnsw_fixture_strategy(), any::<u16>(), any::<u16>()),
            |(fixture, query_hint, k_hint)| {
                run_search_correctness_property(fixture, query_hint, k_hint)
            },
        )
        .map_err(|err| match err {
            TestError::Abort(reason) => {
                TestCaseError::fail(format!("hnsw search proptest aborted: {reason}"))
            }
            TestError::Fail(reason, value) => TestCaseError::fail(format!(
                "hnsw search proptest failed: {reason}; minimal input: {value:#?}"
            )),
        })?;
    Ok(())
}
