//! Property coverage for the excess-of-mass cluster-selection rule.
//!
//! Integer-valued weights make an exact stability tie between a parent cluster
//! and the summed stability of its children routine. The design's
//! excess-of-mass rule selects the parent only when its stability strictly
//! exceeds that total, so a tie must keep the child clusters.
//!
//! The property restates the rule as a recurrence over the condensed forest and
//! compares the extractor's surviving selection set against it. That is the
//! only way to catch a tie resolved towards the parent for inputs no fixed
//! example covers: a tie selects the parent instead of the children, so the
//! two sets differ exactly when the comparison is not strict.

use core::num::NonZeroUsize;

use proptest::prelude::*;

use crate::test_utils::suite_proptest_config;
use crate::{CandidateEdge, EdgeHarvest, parallel_kruskal};

use super::{CondensedCluster, CondensedForest, select_stable_clusters};

/// Number of generated graphs.
///
/// A narrow integer weight range repeats split lambdas often, which is what
/// makes exact ties reachable from randomly generated input.
const CASES: u32 = 256;

/// Smallest dataset that can split into two child clusters at `min = 2`.
const MIN_NODES: usize = 4;

/// Largest dataset kept small enough for a per-case forest walk.
const MAX_NODES: usize = 10;

/// Lowest generated edge weight.
const MIN_WEIGHT: u8 = 1;

/// Highest generated edge weight.
const MAX_WEIGHT: u8 = 3;

/// Smallest population condensation retains as a cluster.
const MIN_CLUSTER_SIZE: usize = 2;

/// A complete graph over `MIN_NODES..=MAX_NODES` points with integer weights.
///
/// A complete graph is always connected, so Kruskal always returns a spanning
/// tree and the condensed forest has exactly one hierarchy to inspect.
fn complete_graph_strategy() -> impl Strategy<Value = (usize, Vec<u8>)> {
    (MIN_NODES..=MAX_NODES).prop_flat_map(|node_count| {
        let pair_count = node_count
            .saturating_mul(node_count.saturating_sub(1))
            .checked_div(2)
            .unwrap_or_default();
        (
            Just(node_count),
            prop::collection::vec(MIN_WEIGHT..=MAX_WEIGHT, pair_count),
        )
    })
}

/// Pair every unordered point pair with one generated weight.
fn harvest_for(node_count: usize, weights: &[u8]) -> EdgeHarvest {
    let pairs =
        (0..node_count).flat_map(|left| ((left + 1)..node_count).map(move |right| (left, right)));
    let mut edges = Vec::new();
    let mut sequence = 0_u64;
    for ((left, right), weight) in pairs.zip(weights.iter().copied()) {
        edges.push(CandidateEdge::new(left, right, f32::from(weight), sequence));
        sequence = sequence.saturating_add(1);
    }
    EdgeHarvest::new(edges)
}

/// Look up a cluster the condensed forest must contain.
fn cluster_at(condensed: &CondensedForest, cluster_id: usize) -> &CondensedCluster {
    condensed
        .clusters
        .get(cluster_id)
        .expect("child identifiers are cluster indices")
}

/// Sum, in child order, of the scores each direct child returns upward.
///
/// The total reproduces the extractor's own `f32` accumulator, so an exact tie
/// compares bit-for-bit rather than within a tolerance.
#[expect(
    clippy::float_arithmetic,
    reason = "the total must mirror the extractor's f32 child accumulator"
)]
fn children_total(condensed: &CondensedForest, cluster_id: usize) -> f32 {
    cluster_at(condensed, cluster_id)
        .children
        .iter()
        .copied()
        .fold(0.0_f32, |total, child| {
            total + returned_score(condensed, child)
        })
}

/// Score the extractor returns for one subtree.
///
/// A selected cluster retains its own stability; an unselected one passes its
/// children's total upward, which is how the extractor propagates a score.
fn returned_score(condensed: &CondensedForest, cluster_id: usize) -> f32 {
    if selects_within_subtree(condensed, cluster_id) {
        cluster_at(condensed, cluster_id).stability
    } else {
        children_total(condensed, cluster_id)
    }
}

/// Whether the rule selects `cluster_id` when only its subtree is considered.
fn selects_within_subtree(condensed: &CondensedForest, cluster_id: usize) -> bool {
    let cluster = cluster_at(condensed, cluster_id);
    cluster.children.is_empty() || cluster.stability > children_total(condensed, cluster_id)
}

/// Whether any strict ancestor of `cluster_id` is itself selected.
///
/// A selected ancestor replaces the descendant selections, so descendants of a
/// selected cluster never survive in the final set.
fn has_selected_ancestor(condensed: &CondensedForest, cluster_id: usize) -> bool {
    let mut current = cluster_at(condensed, cluster_id).parent;
    while let Some(ancestor) = current {
        if selects_within_subtree(condensed, ancestor) {
            return true;
        }
        current = cluster_at(condensed, ancestor).parent;
    }
    false
}

/// The selection set the rule implies for a whole condensed forest.
fn expected_selection(condensed: &CondensedForest) -> Vec<usize> {
    let mut selection: Vec<usize> = (0..condensed.clusters.len())
        .filter(|&cluster_id| {
            selects_within_subtree(condensed, cluster_id)
                && !has_selected_ancestor(condensed, cluster_id)
        })
        .collect();
    selection.sort_unstable();
    selection
}

proptest! {
    #![proptest_config(suite_proptest_config(CASES))]

    /// An exact tie keeps the children, so the selection set always matches the
    /// strict excess-of-mass rule.
    #[test]
    fn eom_selection_keeps_exact_ties_as_children((node_count, weights) in complete_graph_strategy()) {
        let harvest = harvest_for(node_count, &weights);
        let forest = parallel_kruskal(node_count, &harvest)
            .expect("a complete graph is connected and its weights are finite");
        prop_assert!(forest.is_tree(), "a complete graph must yield one spanning tree");

        let min_cluster_size = NonZeroUsize::new(MIN_CLUSTER_SIZE).expect("constant is non-zero");
        let condensed = CondensedForest::from_mst(node_count, forest.edges(), min_cluster_size)
            .expect("a validated spanning tree must condense");
        let mut actual = select_stable_clusters(&condensed)
            .expect("a condensed forest must select clusters");
        actual.sort_unstable();
        prop_assert!(
            !actual.is_empty(),
            "a condensed forest always selects at least one cluster"
        );

        prop_assert_eq!(actual, expected_selection(&condensed));
    }
}
