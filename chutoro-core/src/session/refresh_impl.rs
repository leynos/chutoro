//! Refresh method and refresh metrics for [`super::ClusteringSession`].
//!
//! The pure merge → reweight → Kruskal → extract pipeline lives in
//! [`super::refresh`]; this module supplies the impure adapter step (core
//! distance recompute through HNSW), times the whole call, and performs the
//! single all-or-nothing snapshot publish.
//!
//! The split keeps [`super::session_impl`] below the repository's per-file
//! size cap; the inherent `refresh` method is still available on the session.

use std::sync::Arc;

use super::{
    ClusteringSession,
    refresh::{RefreshInputs, RefreshOutcome, rebuild_mst_labels},
};
#[cfg(feature = "metrics")]
use crate::ChutoroError;
use crate::{DataSource, Result};
use tracing::{debug, instrument, warn};

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    /// Rebuilds and publishes the cluster label snapshot.
    ///
    /// The refresh first recomputes any dirty core distances, including the
    /// existing points disturbed by newly appended ones. It then merges the
    /// raw MST backbone, the retained historical edges, and the pending
    /// candidate edges; reweights that combined set with the
    /// mutual-reachability formula using the current core distances; runs the
    /// shared Kruskal machinery; extracts labels; and retains the raw inputs
    /// selected into the new MST backbone.
    ///
    /// Publication is all-or-nothing: a failed refresh leaves `labels`,
    /// `snapshot_version`, `pending_edges`, and the MST backbone unchanged. A
    /// retry after the cause is removed reprocesses the remaining dirty
    /// core-distance cells and republishes. On success the session clears
    /// `pending_edges` and advances `snapshot_version` by exactly one.
    ///
    /// An empty session publishes an empty snapshot, and a session with fewer
    /// points than `min_cluster_size` publishes an all-noise snapshot; both
    /// advance the version so a warm-up session can make progress. This
    /// differs intentionally from batch [`crate::Chutoro::run`], which errors
    /// on the same inputs.
    ///
    /// Appended point ids must form the contiguous prefix
    /// `[0, point_count)`: HNSW node ids are the source indices, and both core
    /// distances and label extraction are indexed by them.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ChutoroError::DataSource`] when the backing source
    /// rejects a distance query during the core-distance recompute, and
    /// [`crate::ChutoroError::CpuHnswFailure`] when an HNSW search fails
    /// there. From the rebuild, returns [`crate::ChutoroError::CpuMstFailure`]
    /// — with code `INVALID_NODE_ID` for a non-contiguous append, and the
    /// other MST codes for internal or runtime faults — and
    /// [`crate::ChutoroError::CpuHierarchyFailure`] when label extraction
    /// rejects the forest. The users' guide's error tables classify each code.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use chutoro_core::{
    /// #     ChutoroBuilder, ChutoroError, DataSource, DataSourceError,
    /// #     MetricDescriptor,
    /// # };
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    /// #         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    /// #         Ok((a - b).abs())
    /// #     }
    /// #     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("abs") }
    /// # }
    /// # fn main() -> Result<(), ChutoroError> {
    /// let source = Arc::new(Dummy(vec![0.0, 1.0, 2.0, 3.0]));
    /// let mut session = ChutoroBuilder::new()
    ///     .with_min_cluster_size(2)
    ///     .build_session(source)?;
    ///
    /// session.append(&[0, 1, 2, 3])?;
    /// assert_eq!(session.snapshot_version(), 0);
    ///
    /// session.refresh()?;
    /// assert_eq!(session.snapshot_version(), 1);
    /// assert_eq!(session.labels().len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self), level = "debug")]
    pub fn refresh(&mut self) -> Result<()> {
        #[cfg(feature = "metrics")]
        let started = self.clock.now();
        let merged_edges = self.merged_edge_count();

        match self.recompute_and_rebuild() {
            Ok((node_count, outcome)) => {
                #[cfg(feature = "metrics")]
                {
                    metrics::histogram!("chutoro.session.refresh.seconds")
                        .record(self.clock.now().duration_since(started).as_secs_f64());
                    record_refresh_count("chutoro.session.refresh.candidate_edges", merged_edges);
                    record_refresh_count(
                        "chutoro.session.refresh.cluster_count",
                        distinct_label_count(&outcome.labels),
                    );
                }
                self.publish_refresh(node_count, outcome);
                debug!(
                    node_count,
                    merged_edges, "refresh: snapshot published and pending edges cleared"
                );
                Ok(())
            }
            Err(error) => {
                warn!(
                    error = ?error,
                    "refresh failed; published snapshot and pending edges unchanged"
                );
                #[cfg(feature = "metrics")]
                metrics::counter!(
                    "chutoro.session.refresh.errors_total",
                    "reason" => refresh_error_reason(&error)
                )
                .increment(1);
                Err(error)
            }
        }
    }

    /// Recomputes core distances, then rebuilds labels and the raw backbone.
    fn recompute_and_rebuild(&mut self) -> Result<(usize, RefreshOutcome)> {
        self.recompute_core_distances()?;
        let node_count = self.point_count();
        let outcome = rebuild_mst_labels(
            self.source.name(),
            &RefreshInputs {
                node_count,
                mst_backbone: &self.mst_backbone,
                historical_edges: &self.historical_edges,
                pending_edges: &self.pending_edges,
                core_distances: &self.core_distances,
                min_cluster_size: self.config.min_cluster_size(),
            },
        )?;
        Ok((node_count, outcome))
    }

    /// Publishes a completed refresh as one all-or-nothing commit block.
    ///
    /// The version bump saturates. At `u64::MAX` a further successful refresh
    /// therefore republishes the snapshot without advancing the counter, which
    /// is the one case where the per-call increment contract does not hold.
    /// Reaching it needs 2^64 refreshes, and the alternative — a new error
    /// variant for an unreachable state — is forbidden by the plan's interface
    /// tolerance, so the boundary is documented rather than rejected.
    fn publish_refresh(&mut self, node_count: usize, outcome: RefreshOutcome) {
        self.labels = Arc::new(outcome.labels);
        self.mst_backbone = outcome.mst_backbone;
        self.pending_edges.clear();
        self.last_refresh_len = node_count;
        self.snapshot_version = self.snapshot_version.saturating_add(1);
    }

    /// Counts the candidate edges a refresh would merge.
    const fn merged_edge_count(&self) -> usize {
        self.mst_backbone
            .len()
            .saturating_add(self.historical_edges.len())
            .saturating_add(self.pending_edges.len())
    }
}

/// Records a `usize` bucket value on a count histogram.
#[cfg(feature = "metrics")]
fn record_refresh_count(metric: &'static str, count: usize) {
    let recorded = u32::try_from(count).map_or_else(|_| f64::from(u32::MAX), f64::from);
    metrics::histogram!(metric).record(recorded);
}

/// Counts distinct labels without assuming they are contiguous.
#[cfg(feature = "metrics")]
fn distinct_label_count(labels: &[usize]) -> usize {
    labels
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

/// Classify a refresh failure for low-cardinality metrics.
#[cfg(feature = "metrics")]
const fn refresh_error_reason(error: &ChutoroError) -> &'static str {
    match error {
        ChutoroError::DataSource { .. } => "data_source",
        ChutoroError::CpuHnswFailure { .. } => "hnsw_failure",
        ChutoroError::CpuMstFailure { .. } => "mst_failure",
        ChutoroError::CpuHierarchyFailure { .. } => "hierarchy_failure",
        _ => "other",
    }
}
