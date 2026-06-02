//! Tests for dense SIMD point packing and tail padding.

use super::*;
use proptest::prelude::*;

#[rstest]
#[case(vec![RowIndex::new(0), RowIndex::new(2)], vec![vec![1.0, 2.0], vec![2.0, 1.0]])]
#[case(
    vec![RowIndex::new(1), RowIndex::new(0), RowIndex::new(2)],
    vec![vec![4.0, 1.0, 2.0], vec![6.0, 2.0, 1.0]],
)]
fn dense_point_view_packs_structure_of_arrays(
    matrix_3x2: Result<RowMajorMatrix<'static>, DataSourceError>,
    #[case] indices: Vec<RowIndex>,
    #[case] expected_blocks: Vec<Vec<f32>>,
) -> Result<(), DataSourceError> {
    let matrix_3x2 = matrix_3x2?;
    let view = DensePointView::from_row_indices(matrix_3x2, &indices)?;
    assert_eq!(view.point_count(), indices.len());
    assert_eq!(view.padded_point_count(), MAX_SIMD_LANES);
    assert!(view.is_aligned_to(SIMD_ALIGNMENT_BYTES));

    for (dimension_index, expected_prefix) in expected_blocks.into_iter().enumerate() {
        let block = view.coordinate_block(dimension_index);
        assert_eq!(&block[..expected_prefix.len()], expected_prefix.as_slice());
        assert!(
            block[expected_prefix.len()..]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
    Ok(())
}

#[rstest]
#[case(0, 0)]
#[case(1, MAX_SIMD_LANES)]
#[case(15, MAX_SIMD_LANES)]
#[case(16, MAX_SIMD_LANES)]
#[case(17, MAX_SIMD_LANES * 2)]
fn dense_point_view_pads_point_count_to_lane_boundary(
    #[case] point_count: usize,
    #[case] expected_padded_count: usize,
) -> Result<(), DataSourceError> {
    let values = vec![1.0_f32; 17];
    let matrix = RowMajorMatrix::new(
        MatrixValues::new(&values),
        RowCount::new(17),
        Dimension::new(1),
    );
    let indices: Vec<RowIndex> = (0..point_count).map(RowIndex::new).collect();

    let view = DensePointView::from_row_indices(matrix, &indices)?;

    assert_eq!(view.padded_point_count(), expected_padded_count);
    Ok(())
}

#[rstest]
#[case(0, 4, 4, 0)]
#[case(1, 0, 4, 1)]
#[case(4, 0, 4, 4)]
#[case(5, 4, 4, 1)]
#[case(15, 8, 8, 7)]
#[case(16, 8, 8, 8)]
#[case(17, 16, 16, 1)]
#[case(17, 20, 16, 0)]
#[case(17, 32, 16, 0)]
fn lane_output_count_limits_writes_to_logical_points(
    #[case] point_count: usize,
    #[case] offset: usize,
    #[case] lanes: usize,
    #[case] expected: usize,
) {
    assert_eq!(lane_output_count(point_count, offset, lanes), expected);
}

proptest! {
    #[test]
    fn lane_output_count_matches_saturating_tail_formula(
        point_count in any::<usize>(),
        offset in any::<usize>(),
        lanes in any::<usize>(),
    ) {
        let expected = point_count.saturating_sub(offset).min(lanes);

        prop_assert_eq!(lane_output_count(point_count, offset, lanes), expected);
    }
}

#[rstest]
#[case(15)]
#[case(17)]
fn dense_point_view_zero_fills_unused_lanes(
    #[case] point_count: usize,
) -> Result<(), DataSourceError> {
    let values: Vec<f32> = (0..17).map(|value| value as f32 + 1.0).collect();
    let matrix = RowMajorMatrix::new(
        MatrixValues::new(&values),
        RowCount::new(17),
        Dimension::new(1),
    );
    let indices: Vec<RowIndex> = (0..point_count).map(RowIndex::new).collect();

    let view = DensePointView::from_row_indices(matrix, &indices)?;
    let block = view.coordinate_block(0);

    assert!(
        block[point_count..view.padded_point_count()]
            .iter()
            .all(|value| *value == 0.0)
    );
    Ok(())
}

#[rstest]
#[case(0, true)]
#[case(1, true)]
#[case(2, false)]
fn dense_point_view_reports_scalar_fallback_preference(
    #[case] point_count: usize,
    #[case] expected: bool,
) {
    const VALUES: [f32; 6] = [1.0, 2.0, 4.0, 6.0, 2.0, 1.0];
    let matrix = RowMajorMatrix::new(
        MatrixValues::new(&VALUES),
        RowCount::new(3),
        Dimension::new(2),
    );
    let indices: Vec<RowIndex> = (0..point_count).map(RowIndex::new).collect();
    let view = DensePointView::from_row_indices(matrix, &indices).expect("view must build");
    assert_eq!(view.prefers_scalar_fallback(), expected);
}
