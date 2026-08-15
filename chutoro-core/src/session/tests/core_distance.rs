//! Core-distance tests for [`ClusteringSession`].
//!
//! These tests pin the session lifecycle introduced by roadmap item 11.1.4:
//! appends mark core distances dirty, explicit recompute fills them, and full
//! recompute mirrors the batch CPU path.

use std::{collections::BTreeSet, sync::Arc};

use proptest::prelude::*;
use rstest::rstest;

use super::common::{SessionTestSource, make_session, session_builder};
use super::core_distance_support::{
    assert_core_distances_eq, expected_batch_cores, monotonic_append_order, observed_cores,
    unique_sorted,
};
use crate::{
    ChutoroBuilder, DataSource, HnswParams, Neighbour, session::core_distance::recompute_targets,
    test_utils::suite_proptest_config,
};

#[rstest]
fn core_distance_returns_none_before_append(session_builder: ChutoroBuilder) {
    let (session, _) = make_session(session_builder, 4).expect("session must build");

    assert_eq!(session.core_distance(0), None);
}

#[rstest]
fn core_distance_returns_none_after_append_before_recompute(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 4).expect("session must build");

    session.append(&[0, 1, 2, 3]).expect("append must succeed");

    for point in 0..4 {
        assert_eq!(session.core_distance(point), None);
    }
}

#[rstest]
fn recompute_core_distances_clears_dirty_bits(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 4).expect("session must build");

    session.append(&[0, 1, 2, 3]).expect("append must succeed");
    session
        .recompute_core_distances()
        .expect("incremental recompute must succeed");

    for point in 0..4 {
        let core = session
            .core_distance(point)
            .expect("recompute must fill every appended point");
        assert!(core.is_finite());
    }
}

#[rstest]
fn recompute_core_distances_matches_batch_per_point(session_builder: ChutoroBuilder) {
    let hnsw_params = HnswParams::default().with_rng_seed(7);
    let (mut session, source) =
        make_session(session_builder.with_hnsw_params(hnsw_params), 4).expect("session must build");

    session.append(&[0, 1, 2, 3]).expect("append must succeed");
    session
        .recompute_core_distances_full()
        .expect("full recompute must succeed");

    let expected =
        expected_batch_cores(source.as_ref(), &session).expect("batch cores must compute");
    assert_core_distances_eq(&session, &expected);
}

#[rstest]
#[case(6, true, "full recompute with saturated neighbour selection")]
#[case(2, false, "incremental recompute with fallback path")]
fn core_distance_matches_expected_batch_result_for_selection_and_fallback(
    session_builder: ChutoroBuilder,
    #[case] point_count: usize,
    #[case] use_full_recompute: bool,
    #[case] scenario: &str,
) {
    let (mut session, source) = make_session(session_builder.with_min_cluster_size(3), point_count)
        .expect("session must build");

    let points: Vec<usize> = (0..point_count).collect();
    session.append(&points).expect("append must succeed");

    if use_full_recompute {
        session
            .recompute_core_distances_full()
            .expect("full recompute must succeed");
    } else {
        session
            .recompute_core_distances()
            .expect("incremental recompute must succeed");
    }

    let expected =
        expected_batch_cores(source.as_ref(), &session).expect("batch cores must compute");
    assert_eq!(
        session.core_distance(0),
        Some(expected[0]),
        "unexpected core distance for {scenario}"
    );
}

#[rstest]
fn core_distance_empty_neighbour_list_yields_zero(session_builder: ChutoroBuilder) {
    let (mut session, _) =
        make_session(session_builder.with_min_cluster_size(3), 1).expect("session must build");

    session.append(&[0]).expect("append must succeed");
    session
        .recompute_core_distances()
        .expect("incremental recompute must succeed");

    assert_eq!(session.core_distance(0), Some(0.0));
}

#[rstest]
fn recompute_core_distances_recomputes_touched_existing_points(session_builder: ChutoroBuilder) {
    let hnsw_params = HnswParams::new(4, 16)
        .expect("HNSW params must be valid")
        .with_rng_seed(19);
    let (mut session, _) = make_session(
        session_builder
            .with_min_cluster_size(1)
            .with_hnsw_params(hnsw_params),
        5,
    )
    .expect("session must build");

    session
        .append(&[0, 2, 4])
        .expect("first append must succeed");
    session
        .recompute_core_distances()
        .expect("first recompute must succeed");
    let before = session
        .core_distance(0)
        .expect("first recompute must fill point 0");

    session.append(&[1]).expect("second append must succeed");
    session
        .recompute_core_distances()
        .expect("second recompute must succeed");
    let after = session
        .core_distance(0)
        .expect("second recompute must keep point 0 filled");

    assert!(after <= before, "expected {after} <= {before}");
}

#[rstest]
fn core_distance_out_of_range_returns_none(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 1).expect("session must build");

    session.append(&[0]).expect("append must succeed");

    assert_eq!(session.core_distance(session.point_count()), None);
}

#[rstest]
fn recompute_core_distances_full_recomputes_all_points(session_builder: ChutoroBuilder) {
    let (mut session, source) = make_session(session_builder, 10).expect("session must build");

    session
        .append(&(0..8).collect::<Vec<_>>())
        .expect("first append must succeed");
    session
        .recompute_core_distances()
        .expect("incremental recompute must succeed");
    session.append(&[8, 9]).expect("second append must succeed");
    session
        .recompute_core_distances_full()
        .expect("full recompute must succeed");

    let expected =
        expected_batch_cores(source.as_ref(), &session).expect("batch cores must compute");
    assert_core_distances_eq(&session, &expected);
}

#[test]
fn recompute_targets_excludes_query_itself() {
    let neighbours = vec![
        Neighbour {
            id: 1,
            distance: 1.0,
        },
        Neighbour {
            id: 2,
            distance: 2.0,
        },
    ];

    let targets = recompute_targets(&[2], &[&neighbours]);

    assert_eq!(targets, vec![1]);
}

#[test]
fn recompute_targets_unions_existing_neighbours() {
    let first = vec![
        Neighbour {
            id: 0,
            distance: 1.0,
        },
        Neighbour {
            id: 1,
            distance: 2.0,
        },
    ];
    let second = vec![
        Neighbour {
            id: 1,
            distance: 1.0,
        },
        Neighbour {
            id: 3,
            distance: 2.0,
        },
    ];

    let targets = recompute_targets(&[2, 4], &[&first, &second]);

    assert_eq!(targets, vec![0, 1, 3]);
}

proptest! {
    #![proptest_config(suite_proptest_config(64))]

    #[test]
    fn prop_core_distance_monotonically_non_increasing_after_saturation(
        min_cluster_size in 1usize..=4,
        tail in proptest::collection::vec(0usize..24, 0..=20)
            .prop_map(unique_sorted)
    ) {
        let inserted = monotonic_append_order(min_cluster_size, tail);
        let source = Arc::new(SessionTestSource::with_len(24));
        let mut session = ChutoroBuilder::new()
            .with_min_cluster_size(min_cluster_size)
            .build_session(Arc::clone(&source))
            .expect("session must build");
        let mut previous: Vec<Option<f32>> = vec![None; source.len()];

        let mut inserted_set = BTreeSet::new();
        for index in inserted {
            session.append(&[index]).expect("append must succeed");
            inserted_set.insert(index);
            session
                .recompute_core_distances_full()
                .expect("full recompute must succeed");

            if inserted_set.len().saturating_sub(1) < min_cluster_size {
                continue;
            }

            for &point in &inserted_set {
                if let Some(before) = previous[point] {
                    let after = session
                        .core_distance(point)
                        .expect("saturated point must have a core distance");
                    prop_assert!(
                        after <= before,
                        "point {point} core distance increased from {before} to {after}"
                    );
                }
                previous[point] = session.core_distance(point);
            }
        }
    }

    #[test]
    fn prop_recompute_full_matches_batch(
        len in 4usize..=12,
        min_cluster_size in 1usize..=3,
    ) {
        let source = Arc::new(SessionTestSource::with_len(len));
        let mut session = ChutoroBuilder::new()
            .with_min_cluster_size(min_cluster_size)
            .build_session(Arc::clone(&source))
            .expect("session must build");

        session
            .append(&(0..len).collect::<Vec<_>>())
            .expect("append must succeed");
        session
            .recompute_core_distances_full()
            .expect("full recompute must succeed");

        let expected = expected_batch_cores(source.as_ref(), &session)
            .expect("batch cores must compute");
        let observed =
            observed_cores(&session).expect("point must have a recomputed core distance");
        prop_assert_eq!(observed, expected);
    }

    #[test]
    fn prop_incremental_recompute_matches_full_for_first_append(
        len in 1usize..=12,
        min_cluster_size in 1usize..=4,
    ) {
        let source = Arc::new(SessionTestSource::with_len(len));
        let mut session = ChutoroBuilder::new()
            .with_min_cluster_size(min_cluster_size)
            .build_session(Arc::clone(&source))
            .expect("session must build");

        session
            .append(&(0..len).collect::<Vec<_>>())
            .expect("append must succeed");
        session
            .recompute_core_distances()
            .expect("incremental recompute must succeed");
        let incremental =
            observed_cores(&session).expect("point must have a recomputed core distance");

        session
            .recompute_core_distances_full()
            .expect("full recompute must succeed");

        let full = observed_cores(&session).expect("point must have a recomputed core distance");
        prop_assert_eq!(incremental, full);
    }
}
