//! Non-finite dense SIMD backend parity properties.

use proptest::prelude::*;

use super::strategies;
use crate::simd::{DensePointView, semantics::DistanceSemantics};

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

    #[test]
    fn query_point_backends_canonicalise_non_finite_to_nan(
        fixture in strategies::non_finite_query_points_fixture(),
    ) {
        let semantics = DistanceSemantics::default_euclidean();
        let matrix = fixture.matrix();
        let query = matrix
            .row(fixture.query_index())
            .expect("query row must exist because fixture.query_index() is always within bounds");
        let points = DensePointView::from_row_indices(matrix, &fixture.point_indices())
            .expect("point rows must exist because fixture point indices are generated in bounds");
        let entries = super::query_points_entries();
        let mut expected = vec![0.0_f32; points.point_count()];

        semantics.oracle_query_points(query.as_slice(), &points, &mut expected);
        prop_assert!(
            expected.iter().any(|distance| distance.is_nan()),
            "scalar oracle must canonicalise at least one query-to-points output to NaN",
        );
        prop_assert!(!entries.is_empty(), "at least scalar backend must be available");
        for (backend, entry) in entries {
            let mut actual = vec![0.0_f32; points.point_count()];
            entry(query.as_slice(), &points, &mut actual);
            semantics.assert_query_close(&actual, &expected);
            for (index, expected_distance) in expected.iter().enumerate() {
                if expected_distance.is_nan() {
                    prop_assert!(
                        actual[index].is_nan(),
                        "backend={backend:?}, index={index}, actual={}",
                        actual[index],
                    );
                }
            }
        }
    }
}
