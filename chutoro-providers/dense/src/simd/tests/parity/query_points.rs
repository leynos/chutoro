//! Query-to-points dense SIMD backend parity properties.
//!
//! This module checks the finite batched Euclidean distance property: every
//! enabled query-to-points backend must match the scalar oracle produced by
//! [`DistanceSemantics::oracle_query_points`].
//!
//! Inputs come from [`strategies::query_points_fixture`], which builds a
//! row-major matrix with row `0` as the query and rows `1..=point_count` as the
//! selected points. The fixture covers arbitrary finite, duplicate-row and
//! all-zero row patterns across SIMD boundary dimensions.
//!
//! A backend output slice is accepted when
//! [`DistanceSemantics::assert_query_close`] finds every finite distance within
//! the configured epsilon of the scalar query-to-points result.

use proptest::prelude::*;

use super::strategies;
use crate::simd::{DensePointView, semantics::DistanceSemantics};

proptest! {
    #![proptest_config(super::proptest_config(64))]

    #[test]
    fn query_point_backends_match_scalar_oracle(fixture in strategies::query_points_fixture()) {
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
        prop_assert!(!entries.is_empty(), "at least scalar backend must be available");
        for (_backend, entry) in entries {
            let mut actual = vec![0.0_f32; points.point_count()];
            entry(query.as_slice(), &points, &mut actual);
            semantics.assert_query_close(&actual, &expected);
        }
    }
}
