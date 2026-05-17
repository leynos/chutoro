//! Non-finite dense SIMD backend parity properties.

use proptest::prelude::*;

use super::strategies;
use crate::simd::semantics::DistanceSemantics;

proptest! {
    #![proptest_config(super::proptest_config(64))]

    #[test]
    fn pairwise_backends_canonicalise_non_finite_to_nan(
        (left, right) in strategies::non_finite_vector_pair(),
    ) {
        let semantics = DistanceSemantics::default_euclidean();
        let expected = semantics.oracle_pairwise(&left, &right);
        let entries = super::pairwise_entries();

        prop_assert!(expected.is_nan(), "scalar oracle must canonicalise to NaN");
        prop_assert!(!entries.is_empty(), "at least scalar backend must be available");
        for (backend, entry) in entries {
            let actual = entry(&left, &right);
            prop_assert!(
                actual.is_nan(),
                "backend={backend:?}, len={}, actual={actual}",
                left.len(),
            );
        }
    }
}
