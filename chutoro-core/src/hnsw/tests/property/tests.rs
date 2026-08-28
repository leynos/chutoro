//! Property-based tests for the HNSW implementation covering mutation plans
//! (add/delete/reconfigure), search correctness, fixture validation, bootstrap
//! reachability, and the shared proptest runners/helpers used to orchestrate
//! these scenarios.

use proptest::{proptest, test_runner::TestCaseResult};
use rstest::rstest;

use super::{
    graph_topology_tests::{
        run_graph_metadata_consistency_property, run_graph_mst_compatibility_property,
        run_graph_validity_property,
    },
    mutation_property::derive_initial_population,
    strategies::graph_fixture_strategy,
    support::DenseVectorSource,
    test_runner_support::{
        JobKind, ShrinkIterations, StackSize, TestCases, idempotency_cases,
        idempotency_shrink_iters, mutation_cases, mutation_shrink_iters, run_idempotency_test,
        run_mutation_test, run_search_test, search_cases, search_shrink_iters,
        select_idempotency_cases, select_idempotency_shrink_iters, select_mutation_cases,
        select_mutation_cases_for_fork, select_mutation_shrink_iters, select_search_cases,
        select_search_shrink_iters,
    },
    types::HnswParamsSeed,
};
use crate::error::DataSourceError;
use crate::hnsw::HnswError;
use crate::test_utils::suite_proptest_config;
use crate::{CpuHnsw, DataSource};

#[test]
fn dense_vector_source_rejects_inconsistent_rows() {
    let empty_error =
        DenseVectorSource::new("empty", Vec::new()).expect_err("empty data should fail");
    assert_eq!(empty_error, DataSourceError::EmptyData);

    let zero_dimension_error =
        DenseVectorSource::new("zero", vec![vec![]]).expect_err("zero dimension should fail");
    assert_eq!(zero_dimension_error, DataSourceError::ZeroDimension);

    let dimension_mismatch_error =
        DenseVectorSource::new("mismatch", vec![vec![0.0, 1.0], vec![1.0]])
            .expect_err("dimension mismatch must fail");
    assert_eq!(
        dimension_mismatch_error,
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

#[test]
#[ignore = "stress configuration is too expensive for the default test suite"]
fn hnsw_mutations_preserve_invariants_proptest_stress() -> TestCaseResult {
    run_mutation_test(
        TestCases::try_new(640).expect("test cases must be > 0"),
        ShrinkIterations::new(4096),
        StackSize::try_new(32 * 1024 * 1024).expect("stack size must be >= minimum"),
    )
}

#[test]
fn hnsw_search_matches_brute_force_proptest() -> TestCaseResult {
    run_search_test(
        search_cases().expect("test cases must be > 0"),
        search_shrink_iters(),
    )
}

#[test]
fn hnsw_idempotency_preserved_proptest() -> TestCaseResult {
    run_idempotency_test(
        idempotency_cases().expect("test cases must be > 0"),
        idempotency_shrink_iters(),
        StackSize::try_new(96 * 1024 * 1024).expect("stack size must be >= minimum"),
    )
}

#[rstest]
#[case(JobKind::Coverage, 250, 1)]
#[case(JobKind::Standard, 250, 16)]
#[case(JobKind::Standard, 16, 16)]
fn select_idempotency_cases_enforces_coverage_budget(
    #[case] job: JobKind,
    #[case] configured_cases: u32,
    #[case] expected_cases: u32,
) {
    let configured = TestCases::try_new(configured_cases).expect("test cases must be > 0");
    assert_eq!(
        select_idempotency_cases(job, configured),
        TestCases::try_new(expected_cases)
    );
}

#[rstest]
#[case(JobKind::Coverage, 8)]
#[case(JobKind::Standard, 1024)]
fn select_idempotency_shrink_iters_enforces_coverage_budget(
    #[case] job: JobKind,
    #[case] expected_iters: u32,
) {
    assert_eq!(
        select_idempotency_shrink_iters(job),
        ShrinkIterations::new(expected_iters)
    );
}

#[rstest]
#[case(JobKind::Coverage, 250, 4)]
#[case(JobKind::Standard, 250, 64)]
#[case(JobKind::Standard, 64, 64)]
fn select_mutation_cases_enforces_pr_budget(
    #[case] job: JobKind,
    #[case] configured_cases: u32,
    #[case] expected_cases: u32,
) {
    let configured = TestCases::try_new(configured_cases).expect("test cases must be > 0");
    assert_eq!(
        select_mutation_cases(job, configured),
        TestCases::try_new(expected_cases)
    );
}

#[rstest]
#[case(JobKind::Coverage, true, 250, 4)]
#[case(JobKind::Coverage, false, 250, 4)]
#[case(JobKind::Standard, true, 250, 250)]
#[case(JobKind::Standard, false, 250, 64)]
fn select_mutation_cases_preserves_forked_deep_run_budget(
    #[case] job: JobKind,
    #[case] fork: bool,
    #[case] configured_cases: u32,
    #[case] expected_cases: u32,
) {
    let configured = TestCases::try_new(configured_cases).expect("test cases must be > 0");
    assert_eq!(
        select_mutation_cases_for_fork(job, configured, fork),
        TestCases::try_new(expected_cases)
    );
}

#[rstest]
#[case(JobKind::Coverage, 64)]
#[case(JobKind::Standard, 1024)]
fn select_mutation_shrink_iters_enforces_coverage_budget(
    #[case] job: JobKind,
    #[case] expected_iters: u32,
) {
    assert_eq!(
        select_mutation_shrink_iters(job),
        ShrinkIterations::new(expected_iters)
    );
}

#[rstest]
#[case(JobKind::Coverage, 250, 4)]
#[case(JobKind::Standard, 250, 250)]
#[case(JobKind::Standard, 64, 64)]
fn select_search_cases_enforces_coverage_budget(
    #[case] job: JobKind,
    #[case] configured_cases: u32,
    #[case] expected_cases: u32,
) {
    let configured = TestCases::try_new(configured_cases).expect("test cases must be > 0");
    assert_eq!(
        select_search_cases(job, configured),
        TestCases::try_new(expected_cases)
    );
}

#[rstest]
#[case(JobKind::Coverage, 64)]
#[case(JobKind::Standard, 1024)]
fn select_search_shrink_iters_enforces_coverage_budget(
    #[case] job: JobKind,
    #[case] expected_iters: u32,
) {
    assert_eq!(
        select_search_shrink_iters(job),
        ShrinkIterations::new(expected_iters)
    );
}

#[test]
fn hnsw_mutations_preserve_invariants_proptest() -> TestCaseResult {
    run_mutation_test(
        mutation_cases().expect("test cases must be > 0"),
        mutation_shrink_iters(),
        StackSize::try_new(96 * 1024 * 1024).expect("stack size must be >= minimum"),
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
    let vectors =
        bootstrap_uniform_vectors().expect("bootstrap uniform vectors fixture should parse");
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

fn bootstrap_uniform_vectors() -> Result<Vec<Vec<f32>>, serde_json::Error> {
    super::fixtures::load_bootstrap_uniform_vectors_from_fixture()
}

// ============================================================================
// Graph Topology Property Tests
// ============================================================================

proptest! {
    #![proptest_config(suite_proptest_config(256))]

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
