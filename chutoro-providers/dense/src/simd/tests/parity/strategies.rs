//! Proptest strategies for dense SIMD backend parity.

use proptest::prelude::*;

use crate::simd::{Dimension, MatrixValues, RowCount, RowIndex, RowMajorMatrix};

const MAX_DIMENSION: usize = 260;
const MAX_POINT_COUNT: usize = 33;

#[derive(Clone, Debug)]
pub(super) struct QueryPointsFixture {
    values: Vec<f32>,
    rows: usize,
    dimension: usize,
    point_indices: Vec<usize>,
}

impl QueryPointsFixture {
    pub(super) fn matrix(&self) -> RowMajorMatrix<'_> {
        RowMajorMatrix::new(
            MatrixValues::new(&self.values),
            RowCount::new(self.rows),
            Dimension::new(self.dimension),
        )
    }

    pub(super) fn query_index(&self) -> RowIndex {
        RowIndex::new(0)
    }

    pub(super) fn point_indices(&self) -> Vec<RowIndex> {
        self.point_indices
            .iter()
            .copied()
            .map(RowIndex::new)
            .collect()
    }
}

pub(super) fn finite_vector_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    lane_dimension().prop_flat_map(|dimension| {
        let zero_pair = Just((vec![0.0_f32; dimension], vec![0.0_f32; dimension]));
        let duplicate_pair = finite_row(dimension).prop_map(|row| (row.clone(), row));
        let arbitrary_pair = (finite_row(dimension), finite_row(dimension));
        prop_oneof![zero_pair, duplicate_pair, arbitrary_pair]
    })
}

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

fn query_fixture_from_rows(
    rows: usize,
    dimension: usize,
    point_count: usize,
) -> impl Strategy<Value = QueryPointsFixture> {
    finite_values(rows, dimension)
        .prop_map(move |values| query_fixture(values, rows, dimension, point_count))
}

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

fn finite_values(rows: usize, dimension: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(finite_value(), rows * dimension)
}

fn finite_row(dimension: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(finite_value(), dimension)
}

fn finite_value() -> impl Strategy<Value = f32> {
    -1.0_f32..1.0_f32
}

fn non_finite_value() -> impl Strategy<Value = f32> {
    prop_oneof![Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY),]
}
