//! Error-path tests for [`ClusteringSession::refresh`].
//!
//! Two failure modes form the finite partition this suite enumerates: a
//! data-source failure during the core-distance recompute and a non-contiguous
//! point-id sequence reported by the pure rebuild. Both must leave every
//! published snapshot field unchanged, and both must succeed on retry once the
//! cause is removed.

use std::{error::Error, num::NonZeroUsize, sync::Arc, time::Duration};

use rstest::rstest;

use super::common::{make_session, session_builder};
use super::failable_source::{FailableSource, FailureMode};
use crate::{ChutoroBuilder, ChutoroError, ClusteringSession, DistanceCacheConfig, HnswParams};

/// Builds a session whose HNSW configuration makes injected failures visible.
fn build_failable_session(
    source: Arc<FailableSource>,
) -> Result<ClusteringSession<FailableSource>, Box<dyn Error>> {
    let cache_entries = NonZeroUsize::new(1).ok_or("cache size must be non-zero")?;
    let hnsw_params = HnswParams::new(2, 4)?.with_distance_cache_config(
        DistanceCacheConfig::new(cache_entries).with_ttl(Some(Duration::ZERO)),
    );

    Ok(ChutoroBuilder::new()
        .with_min_cluster_size(1)
        .with_hnsw_params(hnsw_params)
        .build_session(source)?)
}

/// Returns the error code carried by a CPU MST failure.
fn mst_failure_code(error: &ChutoroError) -> Option<&str> {
    match error {
        ChutoroError::CpuMstFailure { code, .. } => Some(code.as_ref()),
        _ => None,
    }
}

/// The complete published state an all-or-nothing refresh must preserve.
///
/// Every field is compared by value, not by summary statistic: a length check
/// would miss a refresh that reordered or replaced the pending edges while
/// preserving their count.
struct SnapshotState {
    /// Published snapshot version.
    version: u64,
    /// Published label vector.
    labels: Vec<usize>,
    /// Buffered candidate edges awaiting the next refresh.
    pending_edges: Vec<crate::CandidateEdge>,
    /// Retained raw candidate pool.
    mst_backbone: Vec<crate::CandidateEdge>,
}

impl SnapshotState {
    /// Captures the four observable fields OBL-5 protects.
    fn capture<D: crate::DataSource + Send + Sync>(session: &ClusteringSession<D>) -> Self {
        Self {
            version: session.snapshot_version(),
            labels: session.labels().to_vec(),
            pending_edges: session.pending_edges.clone(),
            mst_backbone: session.mst_backbone.clone(),
        }
    }

    /// Compares the captured state with the session's current state.
    ///
    /// Returns the first field that moved, so the caller's `assert_eq!` at the
    /// `#[rstest]` boundary names it. Comparing whole values rather than
    /// summary statistics is deliberate: a length check would miss a refresh
    /// that reordered or replaced the pending edges while preserving count.
    fn diff<D: crate::DataSource + Send + Sync>(
        &self,
        session: &ClusteringSession<D>,
    ) -> Result<(), String> {
        if session.snapshot_version() != self.version {
            return Err(format!(
                "version moved: {} -> {}",
                self.version,
                session.snapshot_version()
            ));
        }
        if session.labels().to_vec() != self.labels {
            return Err(format!("labels moved: {:?}", session.labels()));
        }
        if session.pending_edges != self.pending_edges {
            return Err(format!("pending edges moved: {:?}", session.pending_edges));
        }
        if session.mst_backbone != self.mst_backbone {
            return Err(format!("MST backbone moved: {:?}", session.mst_backbone));
        }
        Ok(())
    }
}

#[rstest]
fn failed_refresh_preserves_snapshot_and_pending() {
    let source = Arc::new(FailableSource::new(FailureMode::DataSource));
    let mut session = build_failable_session(Arc::clone(&source)).expect("session must build");

    session.append(&[0, 1]).expect("first append must succeed");
    session.refresh().expect("initial refresh must publish");
    let version_before = session.snapshot_version();

    session.append(&[2]).expect("second append must succeed");
    assert!(
        !session.pending_edges.is_empty(),
        "test precondition: the second append must harvest edges"
    );
    let before = SnapshotState::capture(&session);

    source.fail();
    let error = session
        .refresh()
        .expect_err("the injected data-source failure must propagate");

    assert!(
        matches!(error, ChutoroError::DataSource { .. }),
        "expected a data-source error, got {error:?}"
    );
    assert_eq!(before.diff(&session), Ok(()));

    source.recover();
    session
        .refresh()
        .expect("retry after recovery must succeed");

    assert_eq!(session.snapshot_version(), version_before + 1);
    assert_eq!(session.labels().len(), 3);
    assert_eq!(session.pending_edges, Vec::new());
}

#[rstest]
fn contiguity_violation_reports_cpu_mst_failure_and_preserves_state(
    session_builder: ChutoroBuilder,
) {
    let (mut session, _) =
        make_session(session_builder.with_min_cluster_size(2), 4).expect("session must build");

    // Index 2 is skipped: append accepts it, but the rebuild requires the
    // appended ids to be the contiguous prefix [0, 3).
    session
        .append(&[0, 1, 3])
        .expect("a non-contiguous append is accepted");
    let before = SnapshotState::capture(&session);
    let version_before = before.version;

    let error = session
        .refresh()
        .expect_err("a non-contiguous refresh must fail");

    assert_eq!(mst_failure_code(&error), Some("INVALID_NODE_ID"));
    assert_eq!(before.diff(&session), Ok(()));

    // Filling the gap repairs the session; the retry republishes.
    session.append(&[2]).expect("filling the gap must succeed");
    session.refresh().expect("retry must succeed");

    assert_eq!(session.labels().len(), 4);
    assert_eq!(session.snapshot_version(), version_before + 1);
}
