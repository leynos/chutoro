//! Flat cluster-label extraction for a condensed single-linkage forest.
//!
//! Condensation decides which clusters survive; this module turns that
//! hierarchy into the per-point label vector callers consume. It selects the
//! maximum-stability cluster set, numbers the selected clusters contiguously,
//! walks the condensed tree to label each surviving point, and appends one
//! extra label for noise.
//!
//! Selection lives with the forest it scores, in the parent module; the
//! traversal lives here, beside the labelling it exists to serve.

use super::{CondensedEvent, CondensedForest, HierarchyError, select_stable_clusters};

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
    let label_lookup = build_label_lookup(condensed.clusters.len(), &selected_ids)?;
    let labels = label_points(node_count, condensed, &label_lookup)?;

    let cluster_count = selected_ids.len();
    // When `cluster_count == 0`, the returned labels are all `0`, representing
    // the dedicated noise label (there are no clusters to offset it from).
    Ok(labels
        .into_iter()
        .map(|label| label.unwrap_or(cluster_count))
        .collect())
}

/// Selects the surviving clusters and returns their identifiers in order.
///
/// The identifiers are sorted so the labels derived from their positions form
/// a deterministic numbering regardless of the order stability scoring visits
/// them in.
fn selected_cluster_ids(condensed: &CondensedForest) -> Result<Vec<usize>, HierarchyError> {
    let mut selected_ids: Vec<usize> = select_stable_clusters(condensed)?.into_iter().collect();
    selected_ids.sort_unstable();
    Ok(selected_ids)
}

/// Builds the dense cluster-identifier to label mapping used by the traversal.
fn build_label_lookup(
    cluster_count: usize,
    selected_ids: &[usize],
) -> Result<Vec<Option<usize>>, HierarchyError> {
    let mut label_lookup = vec![None; cluster_count];
    for (label, cluster_id) in selected_ids.iter().copied().enumerate() {
        *label_lookup
            .get_mut(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })? = Some(label);
    }
    Ok(label_lookup)
}

/// Labels every point reachable from a condensed root, leaving noise unset.
fn label_points(
    node_count: usize,
    condensed: &CondensedForest,
    label_lookup: &[Option<usize>],
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

        // Reborrow the shared forest through its own lifetime, so the event
        // slice outlives the `&mut self` borrows taken while labelling.
        let condensed = self.condensed;
        let events = condensed
            .clusters
            .get(cluster_id)
            .ok_or(HierarchyError::InvalidClusterReference { cluster_id })?
            .events
            .as_slice();

        for &event in events {
            self.apply_event(event, cluster_label)?;
        }

        Ok(())
    }

    /// Write one event: a point leaf takes the label, a child recurses.
    fn apply_event(
        &mut self,
        event: CondensedEvent,
        cluster_label: Option<usize>,
    ) -> Result<(), HierarchyError> {
        match event {
            CondensedEvent::Point { index, .. } => self.label_point(index, cluster_label),
            CondensedEvent::ChildCluster {
                cluster: child_cluster,
                ..
            } => self.label_cluster(child_cluster, cluster_label),
        }
    }

    /// Write the label for one point leaf, rejecting out-of-range indices.
    fn label_point(
        &mut self,
        index: usize,
        cluster_label: Option<usize>,
    ) -> Result<(), HierarchyError> {
        let node_count = self.labels.len();
        *self
            .labels
            .get_mut(index)
            .ok_or(HierarchyError::InvalidPointReference {
                point_id: index,
                node_count,
            })? = cluster_label;
        Ok(())
    }
}
