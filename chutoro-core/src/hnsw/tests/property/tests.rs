//! Property-based tests for the HNSW implementation covering mutation plans
//! (add/delete/reconfigure), search correctness, fixture validation, bootstrap
//! reachability, and the shared proptest runners/helpers used to orchestrate
//! these scenarios.

use proptest::{prop_assert, prop_assert_eq, proptest, test_runner::TestCaseResult};
use rstest::rstest;

use super::{
    graph_topology_tests::{
        run_graph_metadata_consistency_property, run_graph_mst_compatibility_property,
        run_graph_validity_property,
    },
    mutation_property::derive_initial_population,
    strategies::{graph_fixture_strategy, hnsw_fixture_strategy},
    support::{DenseVectorSource, dot, euclidean_distance, l2_norm},
    test_runner_support::{
        ShrinkIterations, StackSize, TestCases, idempotency_cases, idempotency_shrink_iters,
        run_idempotency_test, run_mutation_test, run_search_test, select_idempotency_cases,
        select_idempotency_shrink_iters,
    },
    types::{DistributionMetadata, HnswParamsSeed, VectorDistribution},
};
use crate::error::DataSourceError;
use crate::hnsw::HnswError;
use crate::{CpuHnsw, DataSource};

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
#[ignore]
fn hnsw_mutations_preserve_invariants_proptest_stress() -> TestCaseResult {
    run_mutation_test(
        TestCases::new(640),
        ShrinkIterations::new(4096),
        StackSize::new(32 * 1024 * 1024),
    )
}

#[test]
fn hnsw_search_matches_brute_force_proptest() -> TestCaseResult {
    run_search_test(TestCases::new(64), ShrinkIterations::new(1024))
}

#[test]
fn hnsw_idempotency_preserved_proptest() -> TestCaseResult {
    run_idempotency_test(
        TestCases::new(idempotency_cases()),
        ShrinkIterations::new(idempotency_shrink_iters()),
        StackSize::new(96 * 1024 * 1024),
    )
}

#[rstest]
#[case(true, 250, 4)]
#[case(false, 250, 250)]
#[case(false, 16, 16)]
fn select_idempotency_cases_enforces_coverage_budget(
    #[case] coverage_job: bool,
    #[case] configured_cases: u32,
    #[case] expected_cases: u32,
) {
    assert_eq!(
        select_idempotency_cases(coverage_job, configured_cases),
        expected_cases
    );
}

#[rstest]
#[case(true, 128)]
#[case(false, 1024)]
fn select_idempotency_shrink_iters_enforces_coverage_budget(
    #[case] coverage_job: bool,
    #[case] expected_iters: u32,
) {
    assert_eq!(
        select_idempotency_shrink_iters(coverage_job),
        expected_iters
    );
}

#[test]
fn hnsw_mutations_preserve_invariants_proptest() -> TestCaseResult {
    run_mutation_test(
        TestCases::new(64),
        ShrinkIterations::new(1024),
        StackSize::new(96 * 1024 * 1024),
    )
}

#[test]
fn bootstrap_uniform_fixture_remains_reachable() {
    let seed = HnswParamsSeed {
        max_connections: 2,
        ef_construction: 2,
        level_multiplier: 0.2,
        max_level: 2,
        rng_seed: 0,
    };
    let params = seed.build().expect("params must be valid");
    let vectors = bootstrap_uniform_vectors();
    let source =
        DenseVectorSource::new("uniform-bootstrap", vectors).expect("fixture must be valid");
    let len = source.len();
    let initial_population = derive_initial_population(19, len);
    assert!(
        initial_population > 0,
        "initial_population must be non-zero to exercise bootstrap"
    );
    let index = CpuHnsw::with_capacity(params, len).expect("capacity must be valid");
    for node in 0..initial_population {
        index
            .insert(node, &source)
            .expect("bootstrap insertion must succeed");
    }

    index.inspect_graph(|graph| {
        for node in 0..initial_population {
            let node_ref = graph.node(node).expect("seeded node should exist");
            assert!(
                !node_ref.neighbours(0).is_empty(),
                "seeded node {node} should expose base neighbours",
            );
        }
    });

    index
        .invariants()
        .check_all()
        .expect("bootstrap should preserve reachability");
}

fn bootstrap_uniform_vectors() -> Vec<Vec<f32>> {
    super::fixtures::load_bootstrap_uniform_vectors_from_fixture()
}

// ============================================================================
// Graph Topology Property Tests
// ============================================================================

proptest! {
    #[test]
    fn generated_graphs_are_valid(fixture in graph_fixture_strategy()) {
        run_graph_validity_property(&fixture)?;
    }

    #[test]
    fn graph_metadata_is_consistent(fixture in graph_fixture_strategy()) {
        run_graph_metadata_consistency_property(&fixture)?;
    }

    #[test]
    fn graphs_are_mst_compatible(fixture in graph_fixture_strategy()) {
        run_graph_mst_compatibility_property(&fixture)?;
    }
}
