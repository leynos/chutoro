//! Unit tests for the mutual-reachability weight formula.
//!
//! The formula is the shared scalar that both the batch pipeline and the
//! session refresh apply to a raw edge. A refresh reweights edges that a
//! previous refresh already weighted, so the only correct input is the raw
//! distance: the weight must be the pointwise maximum of the raw distance and
//! the two current core distances, whatever an earlier call returned. These
//! cases pin the algebraic properties the Verus lemma `OBL-1` proves
//! generally: commutativity in the endpoints' core distances, the lower
//! bound on all three arguments, and idempotence when the result is fed back
//! in place of the raw distance.

use rstest::rstest;

use super::super::mutual_reachability_weight;

#[rstest]
#[case(1.0, 2.0, 0.5, 2.0)]
#[case(0.5, 1.0, 0.25, 1.0)]
#[case(3.0, 1.0, 2.0, 3.0)]
#[case(1.5, 1.5, 1.5, 1.5)]
fn weight_is_the_maximum_of_its_arguments(
    #[case] distance: f32,
    #[case] core_source: f32,
    #[case] core_target: f32,
    #[case] expected: f32,
) {
    assert_eq!(
        mutual_reachability_weight(distance, core_source, core_target),
        expected
    );
}

#[rstest]
#[case(1.0, 2.0, 0.5)]
#[case(0.0, 0.0, 0.0)]
#[case(7.5, 3.25, 6.0)]
fn weight_is_symmetric_in_the_endpoint_core_distances(
    #[case] distance: f32,
    #[case] core_source: f32,
    #[case] core_target: f32,
) {
    assert_eq!(
        mutual_reachability_weight(distance, core_source, core_target),
        mutual_reachability_weight(distance, core_target, core_source)
    );
}

#[rstest]
#[case(1.0, 2.0, 0.5)]
#[case(9.0, 1.0, 1.0)]
#[case(0.25, 8.0, 0.5)]
fn weight_is_at_least_every_argument(
    #[case] distance: f32,
    #[case] core_source: f32,
    #[case] core_target: f32,
) {
    let weight = mutual_reachability_weight(distance, core_source, core_target);

    assert!(
        weight >= distance,
        "weight {weight} below raw distance {distance}"
    );
    assert!(
        weight >= core_source,
        "weight {weight} below source core distance {core_source}"
    );
    assert!(
        weight >= core_target,
        "weight {weight} below target core distance {core_target}"
    );
}

/// Reweighting an already weighted edge from its weight must not raise it.
///
/// This is the property that a weighted-backbone retain-and-reweight design
/// violates: after a core distance falls, `max(previous_weight, current
/// cores)` stays at the stale, higher previous weight. The raw distance is
/// always a fixed point of the formula's first argument, so a repeated call
/// that passes the raw distance is history-independent.
#[rstest]
#[case(1.0, 2.0, 0.5)]
#[case(4.0, 0.25, 3.0)]
fn weight_is_idempotent_when_reapplied_to_its_own_result(
    #[case] distance: f32,
    #[case] core_source: f32,
    #[case] core_target: f32,
) {
    let first = mutual_reachability_weight(distance, core_source, core_target);
    let second = mutual_reachability_weight(first, core_source, core_target);

    assert_eq!(first, second);
}

/// A stale, higher weight cannot be rescued by the formula.
///
/// The counterexample is stated as a test because it is the reason the
/// session retains raw edges: with the raw distance `1.0` and current cores
/// `0.5` and `0.25` the correct weight is `1.0`, whereas feeding a previous
/// weight of `9.0` back in yields `9.0`.
#[test]
fn stale_weight_cannot_be_reduced_by_reweighting_the_weight() {
    let raw = 1.0_f32;
    let core_source = 0.5_f32;
    let core_target = 0.25_f32;
    let stale_weight = 9.0_f32;

    assert_eq!(
        mutual_reachability_weight(raw, core_source, core_target),
        1.0
    );
    assert_eq!(
        mutual_reachability_weight(stale_weight, core_source, core_target),
        9.0
    );
}
