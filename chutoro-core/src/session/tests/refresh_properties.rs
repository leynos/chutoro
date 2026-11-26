//! Property tests for the incremental refresh contract.
//!
//! These suites cover the obligations that an example table cannot: label
//! length across generated append/refresh interleavings (OBL-3), the
//! per-call snapshot-version counter (OBL-4), pending-edge clearing, and
//! determinism. The final property compares a multi-round incremental
//! partition with a batch run on the identical dataset (OBL-6).
//!
//! The generators are total: every generated source length, batch size, and
//! cluster count is usable, so there is no rejection budget.

use std::sync::Arc;

use proptest::prelude::*;

use super::common::session_test_source_with_len;
use crate::test_utils::{NumericTestSource, suite_proptest_config};
use crate::{ChutoroBuilder, ClusterId, ClusteringResult, MetricDescriptor, adjusted_rand_index};

/// Maps batch assignments to the plain label integers the ARI helper takes.
fn batch_labels(result: &ClusteringResult) -> Vec<usize> {
    result
        .assignments()
        .iter()
        .map(|id: &ClusterId| usize::try_from(id.get()).expect("batch label must fit usize"))
        .collect()
}

/// Builds a clustered source of `cluster_count` equally spaced groups.
///
/// Clusters are separated by 100 units while each cluster's own points sit
/// one unit apart, so a cluster's widest span stays below the cross-cluster
/// gap for every generated size. Integer arithmetic keeps the fixture free of
/// float rounding.
///
/// Equal spacing is deliberate, not incidental. The multi-refresh bound
/// measures the approximate incremental MST against batch; unequal cluster
/// densities drive that agreement down (a measured 2-3 round case reached ARI
/// ≈0.69), which is the drift boundary roadmap 11.2.4's `refresh_full` exists
/// to bound rather than something 11.2.1 claims. Keeping the generated domain
/// separated and evenly spaced is what makes the 0.95 bound meaningful here.
fn clustered_values(cluster_count: usize, points_per_cluster: usize) -> Vec<f32> {
    (0..cluster_count)
        .flat_map(|cluster| {
            let base = cluster.saturating_mul(100);
            (0..points_per_cluster).map(move |point| {
                u16::try_from(base.saturating_add(point)).map_or(f32::MAX, f32::from)
            })
        })
        .collect()
}

proptest! {
    #![proptest_config(suite_proptest_config(32))]

    /// Across generated append/refresh interleavings the version advances by
    /// exactly one per refresh, the labels always cover the live point count,
    /// and every refresh drains the pending buffer.
    #[test]
    fn refresh_invariants_hold_across_interleavings(
        source_len in 1usize..=64,
        min_cluster_size in 1usize..=6,
        batch_sizes in prop::collection::vec(0usize..=32, 1..=4),
    ) {
        let source = Arc::new(session_test_source_with_len(source_len));
        let mut session = ChutoroBuilder::new()
            .with_min_cluster_size(min_cluster_size)
            .build_session(source)
            .expect("session must build for any configured size");

        let mut next_index = 0usize;
        let mut expected_version = 0u64;
        for batch_size in batch_sizes {
            let end = next_index.saturating_add(batch_size).min(source_len);
            let batch = (next_index..end).collect::<Vec<_>>();
            session.append(&batch).expect("in-bounds contiguous append must succeed");
            next_index = end;

            session.refresh().expect("refresh must succeed for any session state");
            expected_version = expected_version.saturating_add(1);

            prop_assert_eq!(session.snapshot_version(), expected_version);
            prop_assert_eq!(session.labels().len(), session.point_count());
            prop_assert!(session.pending_edges.is_empty());
        }
    }

    /// Two sessions driven by the same *multi-round* append/refresh sequence
    /// publish identical label vectors, so a snapshot is replayable from its
    /// inputs.
    ///
    /// A single append followed by a single refresh would not exercise the
    /// retained backbone or the warm-up pool, which are the parts a replay
    /// could diverge on. The rounds here are derived from the fixture size, so
    /// several rounds genuinely run and every round appends at least one point.
    #[test]
    fn refresh_is_deterministic_for_identical_inputs(
        source_len in 4usize..=24,
        min_cluster_size in 1usize..=4,
        round_count in 2usize..=3,
    ) {
        let expected_cuts = round_cuts(source_len, round_count);
        let run = || {
            let source = Arc::new(session_test_source_with_len(source_len));
            let mut session = ChutoroBuilder::new()
                .with_min_cluster_size(min_cluster_size)
                .build_session(source)
                .expect("session must build for any configured size");
            let mut start = 0usize;
            for end in round_cuts(source_len, round_count) {
                session
                    .append(&(start..end).collect::<Vec<_>>())
                    .expect("contiguous append must succeed");
                start = end;
                session.refresh().expect("refresh must succeed");
            }
            session.labels().to_vec()
        };

        let first = run();
        let second = run();
        prop_assert_eq!(&first, &second);
        prop_assert_eq!(first.len(), source_len);
        prop_assert!(expected_cuts.len() >= 2, "several rounds must run");
    }

    /// After several append/refresh rounds, the incremental partition still
    /// agrees with a batch run on the final dataset within DES-12.7's bound.
    ///
    /// The plan's domain is 8-64 points across 2-4 clusters with 2-3 append
    /// rounds, and every round must append something: the generator derives
    /// the round boundaries from the fixture size, so no round can append
    /// nothing after the source is exhausted.
    #[test]
    fn multi_refresh_matches_batch_within_ari_bound(
        cluster_count in 2usize..=4,
        points_per_cluster in 4usize..=16,
        round_count in 2usize..=3,
    ) {
        let values = clustered_values(cluster_count, points_per_cluster);
        let total = values.len();

        let cut_points = round_cuts(total, round_count);
        let source = Arc::new(
            NumericTestSource::new("refresh-property", values)
                .using_metric_descriptor(MetricDescriptor::new("refresh-property:abs")),
        );
        let mut session = ChutoroBuilder::new()
            .with_min_cluster_size(2)
            .build_session(Arc::clone(&source))
            .expect("session must build");

        let mut start = 0usize;
        for (index, end) in cut_points.iter().copied().enumerate() {
            prop_assert!(end > start, "round {index} must append at least one point");
            session
                .append(&(start..end).collect::<Vec<_>>())
                .expect("contiguous append must succeed");
            start = end;
            session.refresh().expect("refresh must succeed");
        }
        prop_assert_eq!(start, total, "the rounds must cover the whole fixture");

        let chutoro = ChutoroBuilder::new()
            .with_min_cluster_size(2)
            .build()
            .expect("batch configuration must be valid");
        let batch = chutoro.run(source.as_ref()).expect("batch run must succeed");
        let incremental = session.labels().to_vec();
        let ari = adjusted_rand_index(&incremental, &batch_labels(&batch))
            .expect("label lengths must agree");

        prop_assert!(
            ari >= 0.95,
            "incremental/batch ARI {ari} fell below the 0.95 bound"
        );
    }
}

/// Splits `total` into `rounds` strictly increasing cut points ending at `total`.
///
/// Every round therefore appends at least one point, and the final cut is the
/// whole dataset so the batch oracle compares like with like. A round's share
/// is capped at what is left after reserving one point for every later round:
/// dividing `total` by `rounds` alone can leave a trailing cut equal to its
/// predecessor — four points over three rounds would otherwise yield
/// `[2, 4, 4]` — and the last round would append nothing.
fn round_cuts(total: usize, rounds: usize) -> Vec<usize> {
    let round_count = rounds.clamp(1, total);
    let mut cuts = Vec::with_capacity(round_count);
    let mut previous = 0usize;
    for round in 1..=round_count {
        let remaining_rounds = round_count.saturating_sub(round);
        // Leave one point for each round still to come, then never move
        // backwards.
        let latest = total.saturating_sub(remaining_rounds);
        let ideal = round.saturating_mul(total).div_ceil(round_count);
        let cut = ideal.min(latest).max(previous.saturating_add(1)).min(total);
        cuts.push(cut);
        previous = cut;
    }
    cuts
}

#[cfg(test)]
mod round_cut_tests {
    //! Direct checks on the round splitter used by the differential suites.
    //!
    //! The property tests use `round_cuts` to decide how many points each
    //! append round contributes. A splitter that ever repeats a cut would make
    //! a round append nothing while the surrounding test still claimed several
    //! non-empty rounds, so the guarantee is pinned here rather than inferred
    //! from the properties that consume it.

    use super::round_cuts;

    /// Four points over three rounds was the case that used to yield
    /// `[2, 4, 4]`, leaving the final round empty.
    #[test]
    fn every_round_of_the_three_way_split_appends_a_point() {
        let cuts = round_cuts(4, 3);

        assert_eq!(cuts, vec![2, 3, 4]);
    }

    /// The invariant across the domain the generator uses: strictly
    /// increasing, ending at `total`, and never exceeding it.
    #[test]
    fn cuts_are_strictly_increasing_and_end_at_total() {
        for total in 1usize..=64 {
            for rounds in 2usize..=3 {
                assert_strict_increase(&round_cuts(total, rounds), total, rounds);
            }
        }
    }

    /// Asserts one split's cuts are strictly increasing and end at `total`.
    ///
    /// Extracted so the sweep above stays a two-level loop; the checks
    /// themselves are a flat walk over one split.
    fn assert_strict_increase(cuts: &[usize], total: usize, rounds: usize) {
        assert_eq!(cuts.last().copied().unwrap_or(0), total);
        let mut previous = 0;
        for (index, cut) in cuts.iter().copied().enumerate() {
            assert!(
                cut > previous,
                "round {index} of {rounds} over {total} points repeated a cut: {cuts:?}"
            );
            assert!(cut <= total, "cut {cut} exceeds {total}");
            previous = cut;
        }
    }
}
