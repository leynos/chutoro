//! Assignment validation examples and the independent contiguity model.

use chutoro_core::{ClusterId, ClusteringResult, NonContiguousClusterIds};
use proptest::prelude::*;
use rstest::rstest;
use std::collections::HashSet;

#[rstest]
#[case::single(vec![ClusterId::new(0)], 1)]
#[case::two_clusters(vec![ClusterId::new(0), ClusterId::new(1)], 2)]
fn cluster_count_matches_unique_assignments(
    #[case] assignments: Vec<ClusterId>,
    #[case] expected: usize,
) {
    let result = ClusteringResult::try_from_assignments(assignments)
        .expect("test assignments must be contiguous");
    assert_eq!(result.cluster_count(), expected);
}

#[rstest]
#[case::missing_zero(vec![ClusterId::new(1)], NonContiguousClusterIds::MissingZero, "assignments must start at zero")]
#[case::gap(vec![ClusterId::new(0), ClusterId::new(2)], NonContiguousClusterIds::Gap, "assignments must be contiguous")]
#[case::overflow(vec![ClusterId::new(u64::MAX)], NonContiguousClusterIds::Overflow, "assignments must stay within usize range")]
#[case::duplicate(
    vec![ClusterId::new(0), ClusterId::new(0), ClusterId::new(2)],
    NonContiguousClusterIds::Duplicate,
    "assignments must not duplicate identifiers when clusters are missing",
)]
fn try_from_assignments_validates_contiguity(
    #[case] assignments: Vec<ClusterId>,
    #[case] expected_error: NonContiguousClusterIds,
    #[case] error_message: &str,
) {
    let err = ClusteringResult::try_from_assignments(assignments).expect_err(error_message);
    assert_eq!(err, expected_error);
}

fn model_assignment_validation(
    assignments: &[ClusterId],
) -> Result<usize, NonContiguousClusterIds> {
    if assignments.is_empty() {
        return Ok(0);
    }

    let identifiers: Vec<_> = assignments
        .iter()
        .map(|assignment| assignment.get())
        .collect();
    if identifiers
        .iter()
        .any(|identifier| *identifier >= usize::MAX as u64)
    {
        return Err(NonContiguousClusterIds::Overflow);
    }

    let unique: HashSet<_> = identifiers.iter().copied().collect();
    if !unique.contains(&0) {
        return Err(NonContiguousClusterIds::MissingZero);
    }

    let Some(maximum) = identifiers.iter().copied().max() else {
        return Ok(0);
    };
    let has_gap = (0..=maximum).any(|identifier| !unique.contains(&identifier));
    if !has_gap {
        return Ok(unique.len());
    }

    if unique.len() == identifiers.len() {
        Err(NonContiguousClusterIds::Gap)
    } else {
        Err(NonContiguousClusterIds::Duplicate)
    }
}

proptest! {
    #[test]
    fn try_from_assignments_matches_the_contiguity_model(
        assignment_values in prop::collection::vec(
            prop_oneof![
                8 => 0_u64..=16,
                1 => Just(usize::MAX as u64),
                1 => Just(u64::MAX),
            ],
            0..=24,
        ),
    ) {
        let assignments: Vec<_> = assignment_values
            .iter()
            .copied()
            .map(ClusterId::new)
            .collect();
        let expected = model_assignment_validation(&assignments);
        let actual = ClusteringResult::try_from_assignments(assignments.clone());

        match (actual, expected) {
            (Ok(result), Ok(cluster_count)) => {
                prop_assert_eq!(result.assignments(), assignments.as_slice());
                prop_assert_eq!(result.cluster_count(), cluster_count);
            }
            (Err(actual_error), Err(expected_error)) => {
                prop_assert_eq!(actual_error, expected_error);
            }
            (actual_result, expected_result) => {
                prop_assert!(
                    false,
                    "actual result {actual_result:?} disagrees with model {expected_result:?}"
                );
            }
        }
    }
}
