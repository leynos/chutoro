//! Unit tests for [`ClusteringSession::refresh`] and
//! [`ClusteringSession::labels`], defined in
//! `chutoro-core/src/session/session_impl.rs` and `session/mod.rs`.
//!
//! The cases pin the roadmap 11.2.1 acceptance criteria: a published label
//! vector whose length equals the point count, a snapshot version that
//! advances by exactly one per call, pending-edge clearing, the degenerate
//! empty and warm-up paths, and partition equivalence with the batch pipeline.

use std::sync::Arc;

use pretty_assertions::assert_eq;
use rstest::rstest;

use super::common::{make_session, session_builder};
use crate::test_utils::NumericTestSource;
use crate::{ChutoroBuilder, ClusterId, MetricDescriptor, adjusted_rand_index};

/// Builds a source with two well-separated four-point clusters.
///
/// The separation keeps the differential fixture non-degenerate: an all-noise
/// or single-cluster result scores below 1.0 against the batch oracle.
fn two_cluster_source() -> Arc<NumericTestSource> {
    Arc::new(
        NumericTestSource::new(
            "refresh-clusters",
            vec![0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 23.0],
        )
        .using_metric_descriptor(MetricDescriptor::new("refresh-clusters:abs")),
    )
}

/// Asserts that two partitions are identical up to floating-point rounding.
#[expect(
    clippy::float_arithmetic,
    reason = "derived clustering-quality checks need an absolute delta"
)]
fn assert_partitions_match(ari: f64) {
    let delta = (ari - 1.0_f64).abs();
    assert!(
        delta <= 1.0e-12_f64,
        "ARI {ari} deviates from 1.0 by {delta}"
    );
}

/// Maps batch assignments to the plain label integers the ARI helper takes.
fn batch_labels(result: &crate::ClusteringResult) -> Vec<usize> {
    result
        .assignments()
        .iter()
        .map(|id: &ClusterId| usize::try_from(id.get()).expect("batch label must fit usize"))
        .collect()
}

#[rstest]
fn refresh_publishes_labels_of_point_count_length(session_builder: ChutoroBuilder) {
    let (mut session, _) =
        make_session(session_builder.with_min_cluster_size(2), 4).expect("session must build");

    session.append(&[0, 1, 2, 3]).expect("append must succeed");
    session.refresh().expect("refresh must succeed");

    assert_eq!(session.labels().len(), session.point_count());
    assert_eq!(session.labels().len(), 4);
}

#[rstest]
fn refresh_advances_snapshot_version_by_one(session_builder: ChutoroBuilder) {
    let (mut session, _) =
        make_session(session_builder.with_min_cluster_size(2), 4).expect("session must build");
    session.append(&[0, 1, 2, 3]).expect("append must succeed");

    let before = session.snapshot_version();
    session.refresh().expect("first refresh must succeed");
    let after_first = session.snapshot_version();
    session.refresh().expect("second refresh must succeed");
    let after_second = session.snapshot_version();

    assert_eq!(after_first, before + 1);
    assert_eq!(after_second, after_first + 1);
}

#[rstest]
fn refresh_clears_pending_edges(session_builder: ChutoroBuilder) {
    let (mut session, _) =
        make_session(session_builder.with_min_cluster_size(2), 4).expect("session must build");

    session.append(&[0, 1, 2, 3]).expect("append must succeed");
    assert!(
        !session.pending_edges.is_empty(),
        "test precondition: appending into a populated graph harvests edges"
    );

    session.refresh().expect("refresh must succeed");

    assert_eq!(session.pending_edges, Vec::new());
}

#[rstest]
fn refresh_on_empty_session_publishes_empty_snapshot(session_builder: ChutoroBuilder) {
    let (mut session, _) = make_session(session_builder, 4).expect("session must build");

    session.refresh().expect("an empty session must refresh");

    assert_eq!(*session.labels(), Vec::<usize>::new());
    assert_eq!(session.snapshot_version(), 1);
}

#[rstest]
fn refresh_below_min_cluster_size_publishes_all_noise(session_builder: ChutoroBuilder) {
    let (mut session, _) =
        make_session(session_builder.with_min_cluster_size(5), 4).expect("session must build");
    session.append(&[0, 1, 2, 3]).expect("append must succeed");

    session.refresh().expect("a warm-up session must refresh");

    assert_eq!(session.labels().as_slice(), [0, 0, 0, 0]);
    assert_eq!(session.snapshot_version(), 1);
    assert!(
        !session.mst_backbone.is_empty(),
        "warm-up must retain the harvested candidate pool for the first rebuild"
    );
}

/// Warm-up must not throw away the candidate edges collected so far.
///
/// HNSW harvests an edge only when the later endpoint is inserted, so an
/// old-old edge is never re-harvested. A warm-up refresh that dropped its pool
/// would lose those candidates for good, and the first real refresh would see
/// a strictly poorer candidate set. Crossing the threshold in two steps must
/// therefore produce the same partition as appending everything at once.
#[rstest]
fn crossing_min_cluster_size_retains_warm_up_candidates(session_builder: ChutoroBuilder) {
    let source = two_cluster_source();
    let mut stepped = session_builder
        .clone()
        .with_min_cluster_size(4)
        .build_session(Arc::clone(&source))
        .expect("session must build");

    // Three points is strictly below the threshold of four: warm-up.
    stepped
        .append(&[0, 1, 2])
        .expect("warm-up append must succeed");
    stepped.refresh().expect("warm-up refresh must succeed");
    assert_eq!(stepped.labels().as_slice(), [0, 0, 0]);

    stepped
        .append(&[3, 4, 5, 6, 7])
        .expect("second append must succeed");
    stepped.refresh().expect("crossing refresh must succeed");

    let mut one_step = session_builder
        .with_min_cluster_size(4)
        .build_session(Arc::clone(&source))
        .expect("session must build");
    one_step
        .append(&[0, 1, 2, 3, 4, 5, 6, 7])
        .expect("single append must succeed");
    one_step.refresh().expect("single refresh must succeed");

    let stepped_labels = stepped.labels().to_vec();
    let one_step_labels = one_step.labels().to_vec();
    let ari =
        adjusted_rand_index(&stepped_labels, &one_step_labels).expect("label lengths must agree");

    assert_partitions_match(ari);
    assert_eq!(stepped_labels.len(), 8);
}

#[rstest]
fn refresh_matches_batch_partition(session_builder: ChutoroBuilder) {
    let source = two_cluster_source();
    let mut session = session_builder
        .with_min_cluster_size(2)
        .build_session(Arc::clone(&source))
        .expect("session must build");
    session
        .append(&[0, 1, 2, 3, 4, 5, 6, 7])
        .expect("append must succeed");

    session.refresh().expect("refresh must succeed");

    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .build()
        .expect("batch configuration must be valid");
    let batch = chutoro
        .run(source.as_ref())
        .expect("batch run must succeed");

    let incremental = session.labels().to_vec();
    let batch_labels = batch_labels(&batch);
    let ari = adjusted_rand_index(&incremental, &batch_labels).expect("label lengths must agree");

    assert_partitions_match(ari);
    insta::assert_debug_snapshot!(incremental);
}
