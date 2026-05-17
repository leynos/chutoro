//! Pairwise dense SIMD backend parity properties.

use proptest::prelude::*;

use super::strategies;
use crate::simd::semantics::DistanceSemantics;

proptest! {
    #![proptest_config(super::proptest_config(64))]

    #[test]
    fn pairwise_backends_match_scalar_oracle((left, right) in strategies::finite_vector_pair()) {
        let semantics = DistanceSemantics::default_euclidean();
        let expected = semantics.oracle_pairwise(&left, &right);
        let entries = super::pairwise_entries();

        prop_assert!(!entries.is_empty(), "at least scalar backend must be available");
        for (_backend, entry) in entries {
            let actual = entry(&left, &right);
            semantics.assert_close(actual, expected);
        }
    }
}
