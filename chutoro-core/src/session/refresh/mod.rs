//! Pure incremental MST refresh for [`super::ClusteringSession`].
//!
//! This module owns the domain half of `refresh`: everything between "raw
//! edge buffers in" and "labels and raw MST backbone out". It has no HNSW, I/O,
//! or clock dependency, so the merge, mutual-reachability reweight, Kruskal
//! pass, label extraction, and backbone recovery are unit-, property-, and
//! proof-testable in isolation. The impure core-distance recompute stays in
//! the session methods.
//!
//! The backbone is retained as **raw** [`CandidateEdge`] values rather than
//! weighted [`crate::MstEdge`] values. Reweighting a previously weighted edge
//! would ratchet its weight upward whenever a core distance later falls, so
//! every refresh re-derives the weight from the raw distance and the current
//! core distances.

use std::{collections::BTreeSet, num::NonZeroUsize};

use crate::{
    CandidateEdge, HierarchyConfig, MstEdge, Result,
    cpu_pipeline::{map_cpu_hierarchy_error, map_cpu_mst_error, mutual_reachability_edges},
    mst::{MstError, parallel_kruskal_from_edges},
};

/// Immutable inputs for one pure refresh rebuild.
///
/// The three edge buffers are raw candidates: `mst_backbone` and
/// `historical_edges` come from earlier refreshes, and `pending_edges` from
/// insertions since the last one. They may overlap; duplicate raw edges
/// reweight identically and collapse inside Kruskal's deduplication.
pub(super) struct RefreshInputs<'a> {
    /// Total number of points the label vector must cover.
    pub(super) node_count: usize,
    /// Raw candidate edges retained by the previous refresh.
    ///
    /// Carries the selected MST edges once a real rebuild has run. During
    /// warm-up it instead carries every harvested candidate, because no tree
    /// was selected and the edges must survive to the first real rebuild.
    pub(super) mst_backbone: &'a [CandidateEdge],
    /// Raw non-MST edges retained by earlier refreshes.
    ///
    /// Always empty until roadmap item 11.2.5 populates it; it is read now so
    /// the merge shape does not change when retention arrives.
    pub(super) historical_edges: &'a [CandidateEdge],
    /// Raw edges harvested since the previous refresh.
    pub(super) pending_edges: &'a [CandidateEdge],
    /// Current core distance for every point, indexed by point id.
    pub(super) core_distances: &'a [f32],
    /// Smallest population retained as a cluster.
    pub(super) min_cluster_size: NonZeroUsize,
}

/// The published part of a successful refresh rebuild.
#[derive(Debug)]
pub(super) struct RefreshOutcome {
    /// Flat cluster labels, one per point.
    pub(super) labels: Vec<usize>,
    /// Raw candidate edges selected into the new MST backbone.
    pub(super) mst_backbone: Vec<CandidateEdge>,
}

/// Rebuilds labels and the raw MST backbone from the session's edge buffers.
///
/// The refresh is all-or-nothing: this function returns either a complete
/// outcome or an error, and never mutates its inputs.
///
/// A session with no points publishes an empty snapshot. A session below
/// `min_cluster_size` publishes all-noise labels but returns the merged raw
/// candidates as the backbone, so harvest edges survive warm-up; discarding
/// them would permanently lose old-old connections, which HNSW never
/// re-harvests.
///
/// # Errors
///
/// Returns [`crate::ChutoroError::CpuMstFailure`] with code `INVALID_NODE_ID`
/// when the appended point ids are not the contiguous prefix
/// `[0, node_count)`, because the highest appended id is then a node id at or
/// above `node_count`. Also returns [`crate::ChutoroError::CpuMstFailure`] for
/// the other Kruskal validation failures, and
/// [`crate::ChutoroError::CpuHierarchyFailure`] when label extraction rejects
/// the forest.
///
/// # Examples
///
/// ```rust,ignore
/// let outcome = rebuild_mst_labels("source", &RefreshInputs {
///     node_count: 4,
///     mst_backbone: &[],
///     historical_edges: &[],
///     pending_edges: &edges,
///     core_distances: &[0.5; 4],
///     min_cluster_size: NonZeroUsize::MIN,
/// })?;
/// assert_eq!(outcome.labels.len(), 4);
/// ```
pub(super) fn rebuild_mst_labels(
    source_name: &str,
    inputs: &RefreshInputs<'_>,
) -> Result<RefreshOutcome> {
    ensure_contiguous_point_ids(inputs)?;
    if inputs.node_count == 0 {
        return Ok(RefreshOutcome {
            labels: Vec::new(),
            mst_backbone: Vec::new(),
        });
    }
    let combined = combine_raw_edges(inputs);
    if inputs.node_count < inputs.min_cluster_size.get() {
        // No spanning tree exists yet, so the labels are all noise. The
        // candidate edges harvested during warm-up must still survive: HNSW
        // harvests each edge only when the later endpoint is inserted, so
        // dropping them here would lose old-old connections permanently. They
        // ride in the backbone slot as the candidate pool the first real
        // refresh merges.
        return Ok(RefreshOutcome {
            labels: vec![0; inputs.node_count],
            mst_backbone: combined,
        });
    }

    let reweighted = mutual_reachability_edges(source_name, &combined, inputs.core_distances)?;
    let forest = parallel_kruskal_from_edges(inputs.node_count, &reweighted)
        .map_err(|error| map_cpu_mst_error(&error))?;
    let labels = crate::extract_labels_from_mst(
        inputs.node_count,
        forest.edges(),
        HierarchyConfig::new(inputs.min_cluster_size),
    )
    .map_err(|error| map_cpu_hierarchy_error(&error))?;
    let mst_backbone = recover_raw_backbone(&combined, forest.edges());

    Ok(RefreshOutcome {
        labels,
        mst_backbone,
    })
}

/// Rejects a session whose appended point ids are not a contiguous prefix.
///
/// `core_distances` grows to the highest appended id plus one and HNSW rejects
/// duplicate inserts, so the two lengths agree exactly when the appended ids
/// are `[0, node_count)`. A longer vector means the highest appended id,
/// `len - 1`, is at or above `node_count`; that is precisely the invalid
/// endpoint Kruskal would report, reported here before any reweight work.
fn ensure_contiguous_point_ids(inputs: &RefreshInputs<'_>) -> Result<()> {
    if inputs.core_distances.len() == inputs.node_count {
        return Ok(());
    }
    let highest_appended = inputs.core_distances.len().saturating_sub(1);
    Err(map_cpu_mst_error(&MstError::InvalidNodeId {
        node: highest_appended,
        node_count: inputs.node_count,
    }))
}

/// Concatenates the three raw edge buffers into one candidate sequence.
///
/// Order is `mst_backbone`, then `historical_edges`, then `pending_edges`.
/// No filtering or deduplication happens here: deduplication is Kruskal's
/// job, and the merge must stay an index-preserving concatenation so the
/// raw-backbone recovery can map accepted edges back to their raw inputs.
fn combine_raw_edges(inputs: &RefreshInputs<'_>) -> Vec<CandidateEdge> {
    let capacity = inputs
        .mst_backbone
        .len()
        .saturating_add(inputs.historical_edges.len())
        .saturating_add(inputs.pending_edges.len());
    let mut combined = Vec::with_capacity(capacity);
    combined.extend_from_slice(inputs.mst_backbone);
    combined.extend_from_slice(inputs.historical_edges);
    combined.extend_from_slice(inputs.pending_edges);
    combined
}

/// Recovers the raw inputs whose reweighted forms Kruskal selected.
///
/// Each accepted forest edge is matched to at most one raw candidate by the
/// canonical `(source, target, sequence)` key. Matching at most once keeps a
/// raw edge that appears in two buffers from being retained twice, so the
/// recovered backbone is independent of buffer overlap.
fn recover_raw_backbone(
    combined: &[CandidateEdge],
    forest_edges: &[MstEdge],
) -> Vec<CandidateEdge> {
    let mut pending_keys = forest_edges
        .iter()
        .map(|edge| (edge.source(), edge.target(), edge.sequence()))
        .collect::<BTreeSet<_>>();
    let mut backbone = Vec::with_capacity(pending_keys.len());
    for edge in combined {
        if pending_keys.remove(&canonical_key(*edge)) {
            backbone.push(*edge);
        }
    }
    backbone
}

/// Returns the canonical `(smaller endpoint, larger endpoint, sequence)` key.
const fn canonical_key(edge: CandidateEdge) -> (usize, usize, u64) {
    let (source, target) = if edge.source() <= edge.target() {
        (edge.source(), edge.target())
    } else {
        (edge.target(), edge.source())
    };
    (source, target, edge.sequence())
}

#[cfg(test)]
mod tests;
