//! Property tests for the exact-multiset Kani verdict.
//!
//! The unit tests in `verdict` pin the named cases; these state the invariant
//! they are examples of, over generated sets and orderings: the verdict
//! accepts a report exactly when its successful multiset equals the expected
//! set, and it names the defect that broke it.
//!
//! Property tests live here rather than beside the unit tests because the
//! module they exercise is already close to its 400-line limit.
//!
//! Run via `make test`.

//! Property tests for the exact-multiset comparison.
//!
//! The examples above pin the named cases; these state the invariant they
//! are examples of, over generated sets and orderings: the verdict accepts
//! a report exactly when its successful multiset equals the expected set,
//! and it names the defect that broke it.

use std::collections::BTreeSet;

use chutoro_test_support::ci::kani_suite::output::{HarnessResult, Outcome, SuiteReport, Summary};
use chutoro_test_support::ci::kani_suite::verdict::{VerdictError, evaluate};
use proptest::prelude::*;

/// Return a report whose blocks list `names` with `outcome` and whose
/// summary agrees with them.
fn report(names: &[String]) -> SuiteReport {
    SuiteReport {
        results: names
            .iter()
            .map(|name| HarnessResult {
                name: name.clone(),
                outcome: Outcome::Success,
            })
            .collect(),
        summary: Some(Summary {
            succeeded: names.len(),
            failed: 0,
            total: names.len(),
        }),
        reported_failures: Vec::new(),
    }
}

/// Rotate `names` left by `by`, which varies the order without changing
/// the multiset.
fn rotated(names: &[String], by: usize) -> Vec<String> {
    if names.is_empty() {
        return Vec::new();
    }
    let at = by.checked_rem(names.len()).unwrap_or(0);
    names
        .iter()
        .skip(at)
        .chain(names.iter().take(at))
        .cloned()
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn the_verdict_depends_on_the_multiset_not_the_order(
        names in prop::collection::btree_set("[a-z]{1,12}", 1..6),
        rotation in any::<usize>(),
    ) {
        let expected: BTreeSet<String> = names.iter().cloned().collect();
        let ordered = rotated(&names.into_iter().collect::<Vec<_>>(), rotation);
        prop_assert_eq!(
            evaluate(&expected, &report(&ordered)),
            Ok(()),
            "an exact successful set must pass in any order",
        );
    }

    #[test]
    fn an_extra_harness_is_named_as_unexpected(
        names in prop::collection::btree_set("[a-z]{1,12}", 1..6),
        extra in "[a-z]{13,20}",
    ) {
        let expected: BTreeSet<String> = names.iter().cloned().collect();
        let mut reported: Vec<String> = names.iter().cloned().collect();
        reported.push(extra.clone());
        match evaluate(&expected, &report(&reported)) {
            Err(VerdictError::Unexpected { names: unexpected }) => {
                prop_assert_eq!(unexpected, vec![extra]);
            }
            other => prop_assert!(false, "an extra harness must be unexpected, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_harness_is_named_as_missing(
        names in prop::collection::btree_set("[a-z]{1,12}", 2..6),
        index in any::<prop::sample::Index>(),
    ) {
        let expected: BTreeSet<String> = names.iter().cloned().collect();
        let mut reported: Vec<String> = names.iter().cloned().collect();
        let removed = reported.remove(index.index(reported.len()));
        match evaluate(&expected, &report(&reported)) {
            Err(VerdictError::Missing { names: missing }) => {
                prop_assert_eq!(missing, vec![removed]);
            }
            other => prop_assert!(false, "a missing harness must be named, got {other:?}"),
        }
    }

    #[test]
    fn a_duplicated_harness_is_named_as_duplicate(
        names in prop::collection::btree_set("[a-z]{1,12}", 1..6),
        index in any::<prop::sample::Index>(),
    ) {
        let expected: BTreeSet<String> = names.iter().cloned().collect();
        let mut reported: Vec<String> = names.iter().cloned().collect();
        let len = reported.len();
        let Some(repeated) = reported.get(index.index(len)).cloned() else {
            return Err(TestCaseError::fail("the index must be inside the report"));
        };
        reported.push(repeated.clone());
        match evaluate(&expected, &report(&reported)) {
            Err(VerdictError::Duplicate { names: duplicated }) => {
                prop_assert_eq!(duplicated, vec![repeated]);
            }
            other => prop_assert!(false, "a duplicate must be named, got {other:?}"),
        }
    }

    #[test]
    fn a_summary_that_disagrees_with_the_blocks_is_refused(
        names in prop::collection::btree_set("[a-z]{1,12}", 1..6),
        claimed in 0usize..8,
    ) {
        let expected: BTreeSet<String> = names.iter().cloned().collect();
        let ordered: Vec<String> = names.iter().cloned().collect();
        let mut outcome = report(&ordered);
        outcome.summary = Some(Summary {
            succeeded: claimed,
            failed: 0,
            total: ordered.len(),
        });
        let consistent = claimed == ordered.len();
        let verdict = evaluate(&expected, &outcome);
        if consistent {
            prop_assert_eq!(verdict, Ok(()));
        } else {
            prop_assert!(
                matches!(verdict, Err(VerdictError::SummaryMismatch { .. })),
                "a summary that disagrees with the blocks must be refused, got {verdict:?}",
            );
        }
    }
}
