//! Flat cluster extraction from a condensed hierarchy forest.
//!
//! Stability selection and labelling are kept apart from hierarchy
//! construction so the condensation rules in the parent module stay focused
//! on building the forest.

use core::ops::AddAssign;

use super::{CondensedCluster, CondensedEvent, CondensedForest, HierarchyError};

/// Extracts flat cluster labels from a condensed hierarchy forest.
///
/// Cluster labels are contiguous `usize` identifiers starting at `0`. Noise
/// points are assigned a dedicated label appended after the selected clusters.
///
/// When no clusters are selected (for example when all components are smaller
/// than `min_cluster_size` during condensation), the noise label is `0`.
pub(crate) fn extract_flat_labels(
    node_count: usize,
    condensed: &CondensedForest,
) -> Result<Vec<usize>, HierarchyError> {
    if node_count == 0 {
        return Err(HierarchyError::EmptyDataset);
    }
    if condensed.clusters.is_empty() {
        // No condensed clusters implies every point is noise.
        return Ok(vec![0; node_count]);
    }

    let selected_ids = selected_cluster_ids(condensed)?;
    let label_lookup = build_label_lookup(condensed, &selected_ids)?;
    let labels = label_selected_points(condensed, &label_lookup, node_count)?;

    let cluster_count = selected_ids.len();
    // When `cluster_count == 0`, the returned labels are all `0`, representing
    // the dedicated noise label (there are no clusters to offset it from).
    Ok(labels
        .into_iter()
        .map(|label| label.unwrap_or(cluster_count))
        .collect())
}

/// Select the stable cluster identifiers in ascending order.
fn selected_cluster_ids(condensed: &CondensedForest) -> Result<Vec<usize>, HierarchyError> {
    let selected = select_stable_clusters(condensed)?;
    let mut selected_ids: Vec<usize> = selected.into_iter().collect();
    selected_ids.sort_unstable();
    Ok(selected_ids)
}

/// Map each selected cluster identifier to its contiguous output label.
fn build_label_lookup(
    condensed: &CondensedForest,
    selected_ids: &[usize],
) -> Result<Vec<Option<usize>>, HierarchyError> {
    let mut label_lookup = vec![None; condensed.clusters.len()];
    for (label, cluster_id) in selected_ids.iter().copied().enumerate() {
        let slot = label_lookup
            .get_mut(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?;
        *slot = Some(label);
    }
    Ok(label_lookup)
}

/// Populate one label slot per point, leaving noise points unlabelled.
fn label_selected_points(
    condensed: &CondensedForest,
    label_lookup: &[Option<usize>],
    node_count: usize,
) -> Result<Vec<Option<usize>>, HierarchyError> {
    let mut labels = vec![None; node_count];
    let mut labeller = Labeller::new(condensed, label_lookup, &mut labels);
    for root in condensed.roots.iter().copied() {
        labeller.label_cluster(root, None)?;
    }
    Ok(labels)
}

/// Assigns selected condensed clusters to their point-leaf events.
struct Labeller<'a> {
    /// Condensed hierarchy to traverse.
    condensed: &'a CondensedForest,
    /// Selected label for each cluster, if selected.
    label_lookup: &'a [Option<usize>],
    /// Output label slot for each input point.
    labels: &'a mut [Option<usize>],
}

impl<'a> Labeller<'a> {
    /// Create a labeller over the selected clusters and output slots.
    const fn new(
        condensed: &'a CondensedForest,
        label_lookup: &'a [Option<usize>],
        labels: &'a mut [Option<usize>],
    ) -> Self {
        Self {
            condensed,
            label_lookup,
            labels,
        }
    }

    /// Label a cluster's point events and recursively visit its children.
    fn label_cluster(
        &mut self,
        cluster_id: usize,
        inherited: Option<usize>,
    ) -> Result<(), HierarchyError> {
        let cluster_label = self
            .label_lookup
            .get(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?
            .or(inherited);
        let cluster = self
            .condensed
            .clusters
            .get(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?;

        for event in &cluster.events {
            match *event {
                CondensedEvent::Point { index, .. } => {
                    let node_count = self.labels.len();
                    *self
                        .labels
                        .get_mut(index)
                        .ok_or(HierarchyError::InvalidPointReference {
                            point_id: index,
                            node_count,
                        })? = cluster_label;
                }
                CondensedEvent::ChildCluster {
                    cluster: child_cluster,
                    ..
                } => {
                    self.label_cluster(child_cluster, cluster_label)?;
                }
            }
        }

        Ok(())
    }
}

/// Select the maximum-stability cluster set beneath every condensed root.
fn select_stable_clusters(condensed: &CondensedForest) -> Result<Vec<usize>, HierarchyError> {
    let mut selected = Vec::new();
    for root in condensed.roots.iter().copied() {
        select_stable_clusters_inner(condensed, root, &mut selected)?;
    }
    if selected.is_empty() {
        // Fallback: select all roots to avoid returning only noise for
        // well-formed condensed forests.
        selected.extend(condensed.roots.iter().copied());
    }
    Ok(selected)
}

/// Select stable clusters recursively and return the best subtree score.
fn select_stable_clusters_inner(
    condensed: &CondensedForest,
    cluster_id: usize,
    selected: &mut Vec<usize>,
) -> Result<f32, HierarchyError> {
    let cluster = condensed
        .clusters
        .get(cluster_id)
        .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?;

    let (child_score, child_selected) = select_children(condensed, cluster, selected)?;
    if keeps_current_cluster(cluster, child_score) {
        replace_with_cluster(selected, &child_selected, cluster_id);
        return Ok(cluster.stability);
    }

    Ok(child_score)
}

/// Decide whether a cluster's own stability beats its selected descendants.
///
/// A leaf has no descendants, so it always keeps itself. Otherwise the cluster
/// wins only when it beats the summed child score, which leaves exact ties with
/// the children.
fn keeps_current_cluster(cluster: &CondensedCluster, child_score: f32) -> bool {
    cluster.children.is_empty() || child_score <= cluster.stability
}

/// Recursively select each child and report the selected spans in `selected`.
fn select_children(
    condensed: &CondensedForest,
    cluster: &CondensedCluster,
    selected: &mut Vec<usize>,
) -> Result<(f32, Vec<(usize, usize)>), HierarchyError> {
    let mut child_score = 0.0_f32;
    let mut child_selected = Vec::with_capacity(cluster.children.len());
    for child in &cluster.children {
        let before = selected.len();
        let score = select_stable_clusters_inner(condensed, *child, selected)?;
        child_score.add_assign(score);
        child_selected.push((before, selected.len()));
    }
    Ok((child_score, child_selected))
}

/// Replace the selected child spans with the enclosing cluster.
///
/// Draining in reverse order keeps the earlier spans' indices valid.
fn replace_with_cluster(
    selected: &mut Vec<usize>,
    child_selected: &[(usize, usize)],
    cluster_id: usize,
) {
    for (start, end) in child_selected.iter().copied().rev() {
        selected.drain(start..end);
    }
    selected.push(cluster_id);
}
