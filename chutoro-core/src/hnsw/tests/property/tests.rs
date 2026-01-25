//! Property-based tests for the HNSW implementation covering mutation plans
//! (add/delete/reconfigure), search correctness, fixture validation, bootstrap
//! reachability, and the shared proptest runners/helpers used to orchestrate
//! these scenarios.

use proptest::{
    prelude::any,
    prop_assert, prop_assert_eq, proptest,
    test_runner::{Config, TestCaseError, TestCaseResult, TestError, TestRunner},
};
use rstest::rstest;

/// Runs a property test with the given configuration and strategy.
fn run_proptest<S, F>(config: Config, strategy: S, test_name: &str, property: F) -> TestCaseResult
where
    S: proptest::strategy::Strategy,
    F: Fn(S::Value) -> TestCaseResult,
{
    let mut runner = TestRunner::new(config);
    runner
        .run(&strategy, property)
        .map_err(|err| map_test_error(err, test_name))
}

/// Maps TestError to TestCaseError with formatted messages.
fn map_test_error(err: TestError<impl std::fmt::Debug>, test_name: &str) -> TestCaseError {
    match err {
        TestError::Abort(reason) => TestCaseError::fail(format!("{test_name} aborted: {reason}")),
        TestError::Fail(reason, value) => TestCaseError::fail(format!(
            "{test_name} failed: {reason}; minimal input: {value:#?}"
        )),
    }
}

/// Runs a mutation property test with custom configuration and stack size.
fn run_mutation_proptest_with_stack(config: Config, stack_size: usize) -> TestCaseResult {
    std::thread::Builder::new()
        .name("hnsw-mutation".into())
        .stack_size(stack_size)
        .spawn(move || run_mutation_proptest(config))
        .expect("spawn mutation runner")
        .join()
        .expect("mutation runner panicked")
}

fn run_mutation_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), mutation_plan_strategy()),
        "hnsw mutation proptest",
        |(fixture, plan)| run_mutation_property(fixture, plan),
    )
}

/// Runs a property test with custom configuration parameters and stack size.
fn run_test_with_config<F>(
    cases: u32,
    max_shrink_iters: u32,
    stack_size: usize,
    runner: F,
) -> TestCaseResult
where
    F: FnOnce(Config, usize) -> TestCaseResult,
{
    runner(
        Config {
            cases,
            max_shrink_iters,
            ..Config::default()
        },
        stack_size,
    )
}

/// Runs a mutation property test with custom configuration parameters.
fn run_mutation_test(cases: u32, max_shrink_iters: u32, stack_size: usize) -> TestCaseResult {
    run_test_with_config(
        cases,
        max_shrink_iters,
        stack_size,
        run_mutation_proptest_with_stack,
    )
}

/// Runs an idempotency property test with custom configuration parameters.
fn run_idempotency_test(cases: u32, max_shrink_iters: u32, stack_size: usize) -> TestCaseResult {
    run_test_with_config(
        cases,
        max_shrink_iters,
        stack_size,
        run_idempotency_proptest_with_stack,
    )
}

/// Runs a search property test with custom configuration parameters.
fn run_search_test(cases: u32, max_shrink_iters: u32) -> TestCaseResult {
    run_proptest(
        Config {
            cases,
            max_shrink_iters,
            ..Config::default()
        },
        (hnsw_fixture_strategy(), any::<u16>(), any::<u16>()),
        "hnsw search proptest",
        |(fixture, query_hint, k_hint)| {
            run_search_correctness_property(fixture, query_hint, k_hint)
        },
    )
}

/// Runs an idempotency property test with custom configuration and stack size.
fn run_idempotency_proptest_with_stack(config: Config, stack_size: usize) -> TestCaseResult {
    std::thread::Builder::new()
        .name("hnsw-idempotency".into())
        .stack_size(stack_size)
        .spawn(move || run_idempotency_proptest(config))
        .expect("spawn idempotency runner")
        .join()
        .expect("idempotency runner panicked")
}

fn run_idempotency_proptest(config: Config) -> TestCaseResult {
    run_proptest(
        config,
        (hnsw_fixture_strategy(), idempotency_plan_strategy()),
        "hnsw idempotency proptest",
        |(fixture, plan)| run_idempotency_property(fixture, plan),
    )
}

use super::{
    graph_topology_tests::{
        run_graph_metadata_consistency_property, run_graph_mst_compatibility_property,
        run_graph_validity_property,
    },
    idempotency_property::run_idempotency_property,
    mutation_property::derive_initial_population,
    mutation_property::run_mutation_property,
    search_property::run_search_correctness_property,
    strategies::{
        graph_fixture_strategy, hnsw_fixture_strategy, idempotency_plan_strategy,
        mutation_plan_strategy,
    },
    support::{DenseVectorSource, dot, euclidean_distance, l2_norm},
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
    run_mutation_test(640, 4096, 32 * 1024 * 1024)
}

#[test]
fn hnsw_search_matches_brute_force_proptest() -> TestCaseResult {
    run_search_test(64, 1024)
}

#[test]
fn hnsw_idempotency_preserved_proptest() -> TestCaseResult {
    run_idempotency_test(idempotency_cases(), 1024, 96 * 1024 * 1024)
}

fn idempotency_cases() -> u32 {
    if std::env::var("CI").is_ok() { 64 } else { 16 }
}

#[test]
fn hnsw_mutations_preserve_invariants_proptest() -> TestCaseResult {
    run_mutation_test(64, 1024, 96 * 1024 * 1024)
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
