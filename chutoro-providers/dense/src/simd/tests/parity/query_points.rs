//! Query-to-points dense SIMD backend parity properties.

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
