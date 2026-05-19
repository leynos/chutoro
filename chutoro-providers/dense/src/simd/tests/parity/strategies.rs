//! Proptest strategies for dense SIMD backend parity fixtures.
//!
//! The suite has two fixture families. Vector-pair strategies
//! ([`finite_vector_pair`] and [`non_finite_vector_pair`]) generate direct
//! pairwise inputs, while [`QueryPointsFixture`] strategies
//! ([`query_points_fixture`] and [`non_finite_query_points_fixture`]) generate
//! one query row plus a batch of point rows.
//!
//! Lane dimensions are intentionally hard-coded to `1`, `15`, `16`, `17`, `31`,
//! `32`, `33`, `47`, `48`, `64`, `127`, `128` and `129`. Those sizes exercise
//! exact SIMD-lane multiples, one-before and one-after boundaries, and tail
//! handling across the supported backend widths.
//!
//! Query-to-points fixtures mix three row patterns: arbitrary finite rows cover
//! normal numerical variation, duplicate rows force exact zero distances from
//! repeated values, and all-zero rows exercise the zero-vector policy without
//! relying on randomly generated values to hit that shape.
//!
//! [`QueryPointsFixture`] exposes a [`RowMajorMatrix`], a fixed query index
//! that always selects row `0`, and point indices derived from
//! `1..=point_count`. The generated matrix therefore keeps the query separate
//! from every selected point while preserving a compact row-major layout.

use proptest::prelude::*;

use crate::simd::{Dimension, MatrixValues, RowCount, RowIndex, RowMajorMatrix};

const MAX_DIMENSION: usize = 260;
const MAX_POINT_COUNT: usize = 33;

/// Fixture backing query-to-points parity checks.
///
/// `values` stores the row-major matrix data, `rows` and `dimension` describe
/// its layout, and `point_indices` records the point rows used to construct a
/// `DensePointView` in the parity properties.
#[derive(Clone, Debug)]
pub(super) struct QueryPointsFixture {
    values: Vec<f32>,
    rows: usize,
    dimension: usize,
    point_indices: Vec<usize>,
}

impl QueryPointsFixture {
    /// Builds the row-major matrix view over this fixture's backing store.
    pub(super) fn matrix(&self) -> RowMajorMatrix<'_> {
        RowMajorMatrix::new(
            MatrixValues::new(&self.values),
            RowCount::new(self.rows),
            Dimension::new(self.dimension),
        )
    }

    /// Returns the fixed query row index, which is always row `0`.
    pub(super) fn query_index(&self) -> RowIndex {
        RowIndex::new(0)
    }

    /// Returns the point row indices selected for the query-to-points batch.
    pub(super) fn point_indices(&self) -> Vec<RowIndex> {
        self.point_indices
            .iter()
            .copied()
            .map(RowIndex::new)
            .collect()
    }
}

/// Generates equal-length finite vector pairs for pairwise parity tests.
///
/// The generated pairs cover all-zero, duplicate-row and arbitrary-finite row
/// patterns across the configured lane dimensions.
pub(super) fn finite_vector_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    lane_dimension().prop_flat_map(|dimension| {
        let zero_pair = Just((vec![0.0_f32; dimension], vec![0.0_f32; dimension]));
        let duplicate_pair = finite_row(dimension).prop_map(|row| (row.clone(), row));
        let arbitrary_pair = (finite_row(dimension), finite_row(dimension));
        prop_oneof![zero_pair, duplicate_pair, arbitrary_pair]
    })
}

/// Generates a vector pair with exactly one non-finite lane.
///
/// The strategy starts from a finite pair, then injects one `NaN` or infinity
/// value at a random index in one of the two vectors.
pub(super) fn non_finite_vector_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    lane_dimension().prop_flat_map(|dimension| {
        (
            finite_row(dimension),
            finite_row(dimension),
            0..dimension,
            any::<bool>(),
            non_finite_value(),
        )
            .prop_map(|(mut left, mut right, index, insert_left, non_finite)| {
                if insert_left {
                    left[index] = non_finite;
                } else {
                    right[index] = non_finite;
                }
                (left, right)
            })
    })
}

/// Generates finite query-to-points fixtures across the row-pattern families.
pub(super) fn query_points_fixture() -> impl Strategy<Value = QueryPointsFixture> {
    (lane_dimension(), point_count()).prop_flat_map(|(dimension, point_count)| {
        let rows = point_count + 1;
        prop_oneof![
            query_fixture_from_rows(rows, dimension, point_count),
            duplicate_query_fixture(rows, dimension, point_count),
            zero_query_fixture(rows, dimension, point_count),
        ]
    })
}

/// Generates a query-to-points fixture with exactly one non-finite element.
///
/// The strategy starts from finite values, then replaces one element in either
/// the query row or a selected point row with `NaN` or infinity.
pub(super) fn non_finite_query_points_fixture() -> impl Strategy<Value = QueryPointsFixture> {
    (lane_dimension(), point_count()).prop_flat_map(|(dimension, point_count)| {
        let rows = point_count + 1;
        (
            finite_values(rows, dimension),
            0_usize..=point_count,
            0_usize..dimension,
            non_finite_value(),
        )
            .prop_map(move |(mut values, row_index, column_index, non_finite)| {
                values[row_index * dimension + column_index] = non_finite;
                query_fixture(values, rows, dimension, point_count)
            })
    })
}

/// Generates a fixture from arbitrary finite row-major values.
fn query_fixture_from_rows(
    rows: usize,
    dimension: usize,
    point_count: usize,
) -> impl Strategy<Value = QueryPointsFixture> {
    finite_values(rows, dimension)
        .prop_map(move |values| query_fixture(values, rows, dimension, point_count))
}

/// Generates a fixture whose query and point rows all contain the same values.
fn duplicate_query_fixture(
    rows: usize,
    dimension: usize,
    point_count: usize,
) -> impl Strategy<Value = QueryPointsFixture> {
    finite_row(dimension).prop_map(move |row| {
        let values = std::iter::repeat_n(row, rows).flatten().collect();
        query_fixture(values, rows, dimension, point_count)
    })
}

/// Generates a fixture whose query and point rows are all zero vectors.
fn zero_query_fixture(
    rows: usize,
    dimension: usize,
    point_count: usize,
) -> impl Strategy<Value = QueryPointsFixture> {
    Just(query_fixture(
        vec![0.0_f32; rows * dimension],
        rows,
        dimension,
        point_count,
    ))
}

/// Wraps raw row-major values and point-count metadata in a fixture.
fn query_fixture(
    values: Vec<f32>,
    rows: usize,
    dimension: usize,
    point_count: usize,
) -> QueryPointsFixture {
    QueryPointsFixture {
        values,
        rows,
        dimension,
        point_indices: (1..=point_count).collect(),
    }
}

/// Generates lane dimensions that cover SIMD-width boundaries and tails.
fn lane_dimension() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(1_usize),
        Just(15),
        Just(16),
        Just(17),
        Just(31),
        Just(32),
        Just(33),
        Just(47),
        Just(48),
        Just(64),
        Just(127),
        Just(128),
        Just(129),
        1_usize..=MAX_DIMENSION,
    ]
}

/// Generates the number of selected point rows for query-to-points tests.
fn point_count() -> impl Strategy<Value = usize> {
    prop_oneof![
        Just(1_usize),
        Just(15),
        Just(16),
        Just(17),
        Just(32),
        1_usize..=MAX_POINT_COUNT,
    ]
}

/// Generates row-major finite matrix values for the given shape.
fn finite_values(rows: usize, dimension: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(finite_value(), rows * dimension)
}

/// Generates one finite row with the requested dimension.
fn finite_row(dimension: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(finite_value(), dimension)
}

/// Generates a single finite `f32` value from the parity input range.
fn finite_value() -> impl Strategy<Value = f32> {
    -1.0_f32..1.0_f32
}

/// Generates one non-finite `f32` value used to exercise canonicalisation.
fn non_finite_value() -> impl Strategy<Value = f32> {
    prop_oneof![Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY),]
}
