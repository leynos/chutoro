use proptest::{
    prelude::any,
    prop_assert, prop_assert_eq, proptest,
    test_runner::{Config, TestCaseError, TestCaseResult, TestError, TestRunner},
};
use rstest::rstest;

use super::{
    mutation_property::derive_initial_population,
    mutation_property::run_mutation_property,
    search_property::run_search_correctness_property,
    strategies::{hnsw_fixture_strategy, mutation_plan_strategy},
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
    std::thread::Builder::new()
        .name("hnsw-mutation-stress".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let mut runner = TestRunner::new(Config {
                cases: 640,
                max_shrink_iters: 4096,
                ..Config::default()
            });
            runner
                .run(
                    &(hnsw_fixture_strategy(), mutation_plan_strategy()),
                    |(fixture, plan)| run_mutation_property(fixture, plan),
                )
                .map_err(|err| match err {
                    TestError::Abort(reason) => {
                        TestCaseError::fail(format!("hnsw mutation proptest aborted: {reason}",))
                    }
                    TestError::Fail(reason, value) => TestCaseError::fail(format!(
                        "hnsw mutation proptest failed: {reason}; minimal input: {value:#?}",
                    )),
                })
        })
        .expect("spawn stress runner")
        .join()
        .expect("stress runner panicked")
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

#[test]
fn hnsw_mutations_preserve_invariants_proptest() -> TestCaseResult {
    let mut runner = TestRunner::new(Config {
        cases: 64,
        max_shrink_iters: 2048,
        ..Config::default()
    });
    runner
        .run(
            &(hnsw_fixture_strategy(), mutation_plan_strategy()),
            |(fixture, plan)| run_mutation_property(fixture, plan),
        )
        .map_err(|err| match err {
            TestError::Abort(reason) => {
                TestCaseError::fail(format!("hnsw mutation proptest aborted: {reason}"))
            }
            TestError::Fail(reason, value) => TestCaseError::fail(format!(
                "hnsw mutation proptest failed: {reason}; minimal input: {value:#?}"
            )),
        })?;
    Ok(())
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
    let vectors = vec![
        vec![
            -0.396_401_4,
            -0.574_703_04,
            -1.152_957_6,
            0.531_228_8,
            0.382_021_3,
            0.262_859_23,
            0.918_031_8,
            -0.216_050_39,
        ],
        vec![
            0.155_689_36,
            -1.092_277_3,
            -0.479_329_05,
            1.093_218_7,
            -0.502_325_24,
            -1.234_767_7,
            -0.783_448_5,
            -1.039_418_3,
        ],
        vec![
            0.900_763_4,
            -0.246_779_44,
            1.321_268_2,
            1.240_773_6,
            0.638_659_24,
            0.297_954_56,
            -1.118_238_9,
            0.910_250_07,
        ],
        vec![
            -1.008_921_9,
            -0.862_424,
            0.610_633_13,
            -0.781_000_55,
            -0.293_144_46,
            0.418_012_74,
            1.209_498_5,
            0.750_078_3,
        ],
        vec![
            -1.045_952_4,
            0.563_875_2,
            0.111_062_765,
            -0.381_774_37,
            0.723_139_17,
            -0.119_114_995,
            -0.267_742_04,
            0.925_248_74,
        ],
        vec![
            0.048_650_265,
            1.342_758,
            -1.177_938_5,
            0.876_737_5,
            0.561_164_5,
            1.305_7,
            1.189_487_8,
            1.003_763_8,
        ],
        vec![
            -0.771_407_25,
            0.109_181_05,
            -0.472_181_08,
            1.142_654_5,
            0.335_322_5,
            -1.111_399_8,
            0.257_928_6,
            -0.765_718_3,
        ],
        vec![
            -0.783_968_9,
            -1.260_217_7,
            0.098_876_24,
            -0.426_839_1,
            0.833_862_4,
            -0.796_487_5,
            0.955_031_04,
            0.462_075_7,
        ],
        vec![
            -0.929_420_2,
            1.347_091_3,
            0.806_613_3,
            0.422_841_9,
            0.315_650_82,
            0.444_848_78,
            -0.834_046_2,
            -1.300_822,
        ],
        vec![
            -1.340_137_1,
            -0.085_052_97,
            0.625_077_96,
            -1.034_425_4,
            -0.369_696_14,
            0.951_833_6,
            0.825_142,
            -1.250_152_5,
        ],
        vec![
            0.388_505_94,
            1.075_180_2,
            0.942_937,
            0.684_565_2,
            0.995_674_25,
            -0.023_818_135,
            -0.763_824_4,
            -0.046_542_287,
        ],
        vec![
            -1.220_171_8,
            -0.246_241_81,
            1.058_289_4,
            1.085_396_6,
            -0.618_380_5,
            -1.211_029_4,
            0.843_734_15,
            1.008_184_8,
        ],
        vec![
            0.453_891_52,
            0.986_010_2,
            -0.949_058_8,
            -1.271_146_9,
            -1.280_239_1,
            -0.256_128_67,
            0.554_179_8,
            -0.119_055_27,
        ],
        vec![
            0.867_020_7,
            -0.885_747_2,
            -0.820_472_7,
            -0.772_429_65,
            -1.103_221_3,
            -1.251_388_8,
            -1.333_986_3,
            1.106_944_7,
        ],
    ];

    let source =
        DenseVectorSource::new("uniform-bootstrap", vectors).expect("fixture must be valid");
    let len = source.len();
    let initial_population = derive_initial_population(19, len);
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
