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

use std::{collections::BTreeMap, collections::BTreeSet, num::NonZeroUsize};

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
            mst_backbone: dedup_raw_candidates(combined),
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
    let mst_backbone = recover_raw_backbone(&combined, &reweighted, forest.edges());

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
/// canonical `(source, target, sequence)` key **and** by its reweighted weight.
/// When several candidates satisfy both — which happens whenever a core
/// distance dominates their raw distances, so distinct raw distances reweight
/// to the same weight — the one with the smallest raw distance wins. That rule
/// is independent of buffer order, so duplicating an input anywhere in the
/// merge cannot change the recovered backbone (OBL-7).
///
/// The output is then sorted by canonical key and raw distance, so even the
/// element *order* is independent of which buffer supplied a candidate. The
/// backbone is an unordered set semantically — the next refresh merges it again
/// — so imposing an order costs nothing and makes the snapshot reproducible.
///
/// The weight is part of the match because the endpoint-and-sequence key alone
/// can be ambiguous: two directly constructed candidates may share a key while
/// carrying different raw distances, as in `(0, 1, 9, 7)` and `(0, 1, 1, 7)`.
fn recover_raw_backbone(
    combined: &[CandidateEdge],
    reweighted: &[CandidateEdge],
    forest_edges: &[MstEdge],
) -> Vec<CandidateEdge> {
    // Bucket candidates by the exact accepted identity so lookup is O(log n)
    // rather than a scan per forest edge. `to_bits` gives a total, exact key
    // for the weight; the accepted weight was computed by the same transform
    // from the same inputs, so the bits agree.
    let mut candidates: CandidateIndex = BTreeMap::new();
    for (index, (edge, weighted)) in combined.iter().zip(reweighted).enumerate() {
        candidates
            .entry((canonical_key(*edge), weighted.distance().to_bits()))
            .or_default()
            .push(index);
    }

    let mut claimed = BTreeSet::new();
    let mut backbone = Vec::with_capacity(forest_edges.len());
    for accepted in forest_edges {
        let bucket_key = (
            (accepted.source(), accepted.target(), accepted.sequence()),
            accepted.weight().to_bits(),
        );
        let Some(bucket) = candidates.get(&bucket_key) else {
            continue;
        };
        let chosen = bucket
            .iter()
            .copied()
            .filter(|index| !claimed.contains(index))
            .min_by(|first, second| {
                raw_distance_at(combined, *first).total_cmp(&raw_distance_at(combined, *second))
            });
        if let Some(index) = chosen {
            claimed.insert(index);
            if let Some(edge) = combined.get(index) {
                // Canonical orientation keeps the backbone's element equality
                // independent of which buffer supplied the candidate; the edge
                // is undirected, so the orientation carries no meaning.
                backbone.push(edge.canonicalize());
            }
        }
    }
    sort_raw_candidates(&mut backbone);
    backbone
}

/// Drops repeated raw candidates and imposes a deterministic order.
///
/// A warm-up refresh retains the merged pool rather than a tree, so it must
/// apply the same overlap tolerance the real rebuild gets from Kruskal's
/// deduplication. Otherwise the same raw edge reaching the merge through two
/// buffers would be retained twice and the pool would grow with each warm-up
/// refresh. Identity is the canonical `(source, target, sequence)` key plus the
/// raw distance, so two same-key candidates that differ only in distance both
/// survive; Kruskal decides between them at the first real rebuild.
///
/// Retained edges are stored in canonical orientation (`source <= target`).
/// The edge is undirected, so orientation carries no meaning, but it *is* part
/// of `CandidateEdge`'s derived equality. Without canonicalizing, the pool
/// would keep whichever orientation happened to arrive first, and duplicating
/// the opposite-oriented copy into an earlier buffer would change the pool —
/// a violation of OBL-7 on element equality.
///
/// The retained pool is sorted by the same rule the rebuild uses, so a
/// duplicate in an earlier buffer cannot reorder it either.
fn dedup_raw_candidates(combined: Vec<CandidateEdge>) -> Vec<CandidateEdge> {
    let mut seen = BTreeSet::new();
    let mut retained = Vec::with_capacity(combined.len());
    for edge in combined {
        let canonical = edge.canonicalize();
        if seen.insert((canonical_key(canonical), canonical.distance().to_bits())) {
            retained.push(canonical);
        }
    }
    sort_raw_candidates(&mut retained);
    retained
}

/// Returns a merged candidate's raw distance, or infinity when absent.
///
/// Only used as an ordering key while choosing among tied candidates, so an
/// out-of-range index must sort last rather than panic.
fn raw_distance_at(combined: &[CandidateEdge], index: usize) -> f32 {
    combined
        .get(index)
        .map_or(f32::INFINITY, CandidateEdge::distance)
}

/// Orders raw candidates by canonical key, then raw distance.
///
/// Both terms are total orders over the values the session can hold, so the
/// result is reproducible and independent of merge-buffer order.
fn sort_raw_candidates(edges: &mut [CandidateEdge]) {
    edges.sort_unstable_by(|left, right| {
        canonical_key(*left)
            .cmp(&canonical_key(*right))
            .then_with(|| left.distance().total_cmp(&right.distance()))
    });
}

/// Raw candidates grouped by the exact identity Kruskal accepts.
///
/// The key is the canonical `(source, target, sequence)` plus the accepted
/// weight's bit pattern; the value lists the merged indices carrying it.
type CandidateIndex = BTreeMap<((usize, usize, u64), u32), Vec<usize>>;

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
