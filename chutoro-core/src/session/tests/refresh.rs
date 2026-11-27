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

/// Reports how far `ari` sits from an exact partition match.
///
/// Returns a printable explanation instead of asserting, so the `#[rstest]`
/// boundary owns the assertion and a failure names the deviation.
#[expect(
    clippy::float_arithmetic,
    reason = "derived clustering-quality checks need an absolute delta"
)]
fn partition_mismatch(ari: f64) -> Result<(), String> {
    let delta = (ari - 1.0_f64).abs();
    if delta <= 1.0e-12_f64 {
        Ok(())
    } else {
        Err(format!("ARI {ari} deviates from 1.0 by {delta}"))
    }
}

/// Maps batch assignments to the plain label integers the ARI helper takes.
///
/// Returns the offending label when it does not fit `usize`, so the caller
/// reports it rather than a mapping closure panicking.
fn batch_labels(result: &crate::ClusteringResult) -> Result<Vec<usize>, u64> {
    result
        .assignments()
        .iter()
        .map(|id: &ClusterId| usize::try_from(id.get()).map_err(|_| id.get()))
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

/// OBL-3's four named partitions, on the public session surface.
///
/// The pure rebuild suite pins the exact warm-up label values; these cases pin
/// the same partitions through `append` + `refresh` + `labels`, which is what
/// a caller observes. Only the degenerate size is asserted by label value,
/// because a rebuild at or above the threshold must be allowed to choose any
/// valid partition rather than a hard-coded one.
#[rstest]
#[case::empty_session(0, 2)]
#[case::below_threshold(3, 4)]
#[case::exactly_at_threshold(4, 4)]
#[case::above_threshold(5, 4)]
fn refresh_publishes_a_partition_covering_every_point(
    session_builder: ChutoroBuilder,
    #[case] source_len: usize,
    #[case] min_cluster_size: usize,
) {
    let (mut session, _) = make_session(session_builder.with_min_cluster_size(min_cluster_size), 8)
        .expect("session must build");
    session
        .append(&(0..source_len).collect::<Vec<_>>())
        .expect("contiguous append must succeed");

    session.refresh().expect("refresh must succeed");

    let labels = session.labels();
    assert_eq!(labels.len(), source_len);
    assert_eq!(session.point_count(), source_len);
    assert_eq!(session.snapshot_version(), 1);
    if source_len == 0 {
        assert!(labels.is_empty(), "an empty session publishes no labels");
    }
    if source_len < min_cluster_size {
        assert_eq!(
            labels.as_slice(),
            vec![0; source_len],
            "a below-threshold session publishes all noise"
        );
    }
    assert_eq!(
        session.mst_backbone.is_empty(),
        source_len == 0,
        "a non-empty session retains either a forest or its warm-up pool"
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
    assert_eq!(
        stepped_labels.len(),
        one_step_labels.len(),
        "both sessions must label every point"
    );
    let ari =
        adjusted_rand_index(&stepped_labels, &one_step_labels).expect("label lengths must agree");

    assert_eq!(partition_mismatch(ari), Ok(()));
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
    let batch_labels = batch_labels(&batch).expect("a batch label must fit usize");
    assert_eq!(
        incremental.len(),
        batch_labels.len(),
        "the incremental and batch runs must label the same points"
    );
    let ari = adjusted_rand_index(&incremental, &batch_labels).expect("label lengths must agree");

    assert_eq!(partition_mismatch(ari), Ok(()));
    insta::assert_debug_snapshot!(incremental);
}

/// The version counter saturates instead of wrapping at its limit.
///
/// Reaching `u64::MAX` needs 2^64 refreshes, so the test sets the counter
/// directly; the field is visible to crate tests. A saturated refresh must
/// still publish labels and drain the pending buffer — only the increment is
/// skipped.
#[rstest]
fn refresh_saturates_snapshot_version_at_the_representational_limit(
    session_builder: ChutoroBuilder,
) {
    let (mut session, _) =
        make_session(session_builder.with_min_cluster_size(2), 4).expect("session must build");
    session.append(&[0, 1, 2, 3]).expect("append must succeed");
    session.snapshot_version = u64::MAX;

    session
        .refresh()
        .expect("a refresh at the representational limit must succeed");

    assert_eq!(session.snapshot_version(), u64::MAX);
    assert_eq!(session.labels().len(), 4);
    assert_eq!(session.pending_edges, Vec::new());
}
