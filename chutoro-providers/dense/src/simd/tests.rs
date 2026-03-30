//! Tests for SIMD-aware Euclidean distance kernels.

use super::dispatch::{self, CompiledSimdSupport, RuntimeSimdSupport};
use super::kernels;
use super::*;
use rstest::{fixture, rstest};

mod entrypoints;
mod support_masks;

fn close(left: Distance, right: Distance) {
    let left = left.get();
    let right = right.get();
    let tolerance = 1.0e-6_f32;
    assert!(
        (left - right).abs() <= tolerance,
        "left={left}, right={right}, tolerance={tolerance}",
    );
}

#[fixture]
fn matrix_3x2() -> Result<RowMajorMatrix<'static>, DataSourceError> {
    const VALUES: [f32; 6] = [1.0, 2.0, 4.0, 6.0, 2.0, 1.0];
    Ok(RowMajorMatrix::new(
        MatrixValues::new(&VALUES),
        RowCount::new(3),
        Dimension::new(2),
    ))
}

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

#[rstest]
#[case(
    CompiledSimdSupport::new(false, false, false, false),
    RuntimeSimdSupport::new(true, true, true, true),
    dispatch::EuclideanBackend::Scalar
)]
#[case(
    CompiledSimdSupport::new(true, false, false, false),
    RuntimeSimdSupport::new(true, false, false, false),
    dispatch::EuclideanBackend::Avx2
)]
#[case(
    CompiledSimdSupport::new(true, true, false, false),
    RuntimeSimdSupport::new(true, true, false, false),
    dispatch::EuclideanBackend::Avx512
)]
#[case(
    CompiledSimdSupport::new(false, false, true, false),
    RuntimeSimdSupport::new(false, false, true, false),
    dispatch::EuclideanBackend::Neon
)]
#[case(
    CompiledSimdSupport::new(false, false, false, true),
    RuntimeSimdSupport::new(false, false, false, true),
    dispatch::EuclideanBackend::PortableSimd
)]
#[case(
    CompiledSimdSupport::new(true, false, false, true),
    RuntimeSimdSupport::new(true, false, false, true),
    dispatch::EuclideanBackend::Avx2
)]
#[case(
    CompiledSimdSupport::new(false, true, false, false),
    RuntimeSimdSupport::new(false, false, false, false),
    dispatch::EuclideanBackend::Scalar
)]
#[case(
    CompiledSimdSupport::new(false, false, true, false),
    RuntimeSimdSupport::new(false, false, false, false),
    dispatch::EuclideanBackend::Scalar
)]
#[case(
    CompiledSimdSupport::new(false, false, false, true),
    RuntimeSimdSupport::new(false, false, false, false),
    dispatch::EuclideanBackend::Scalar
)]
fn choose_euclidean_backend_prefers_best_enabled_supported_backend(
    #[case] compiled: CompiledSimdSupport,
    #[case] runtime: RuntimeSimdSupport,
    #[case] expected: dispatch::EuclideanBackend,
) {
    assert_eq!(
        dispatch::choose_euclidean_backend(compiled, runtime),
        expected
    );
}

#[rstest]
#[case(dispatch::EuclideanBackend::Scalar, 2, 2, false)]
#[case(dispatch::EuclideanBackend::Scalar, 2, 0, false)]
#[case(dispatch::EuclideanBackend::Avx2, 2, 0, false)]
#[case(dispatch::EuclideanBackend::Avx2, 2, 1, false)]
#[case(dispatch::EuclideanBackend::Avx512, 2, 1, false)]
#[case(dispatch::EuclideanBackend::Neon, 2, 1, false)]
#[case(dispatch::EuclideanBackend::Avx2, 0, 2, false)]
#[case(dispatch::EuclideanBackend::Avx2, 2, 2, true)]
#[case(dispatch::EuclideanBackend::Avx512, 2, 2, true)]
#[case(dispatch::EuclideanBackend::Neon, 2, 2, true)]
#[case(dispatch::EuclideanBackend::PortableSimd, 2, 2, true)]
#[case(dispatch::EuclideanBackend::PortableSimd, 2, 1, false)]
fn query_point_packing_requires_simd_backend(
    #[case] backend: dispatch::EuclideanBackend,
    #[case] dimension: usize,
    #[case] candidate_count: usize,
    #[case] expected: bool,
) {
    assert_eq!(
        should_pack_query_points_for_backend(backend, dimension, candidate_count),
        expected
    );
}

#[rstest]
#[case(vec![0.0], vec![1.0])]
#[case(vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 8.0])]
#[case(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5.0, 4.0, 3.0, 2.0, 1.0])]
#[case(
    vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
    vec![2.0, 3.0, 5.0, 9.0, 17.0, 33.0, 65.0, 129.0, 257.0],
)]
fn euclidean_distance_matches_scalar_reference(#[case] left: Vec<f32>, #[case] right: Vec<f32>) {
    let expected = kernels::euclidean_distance_scalar(&left, &right);
    let actual = euclidean_distance(RowSlice::new(&left), RowSlice::new(&right));
    close(actual, Distance::new(expected));
}

#[rstest]
#[case(vec![f32::NAN, 1.0], vec![0.0, 0.0])]
#[case(vec![f32::INFINITY, 1.0], vec![0.0, 0.0])]
#[case(vec![f32::NEG_INFINITY, 1.0], vec![0.0, 0.0])]
#[case(vec![1.0, 0.0], vec![f32::NAN, 0.0])]
fn euclidean_distance_canonicalizes_non_finite_inputs_to_nan(
    #[case] left: Vec<f32>,
    #[case] right: Vec<f32>,
) {
    let scalar = kernels::euclidean_distance_scalar(&left, &right);
    let wrapped = euclidean_distance(RowSlice::new(&left), RowSlice::new(&right)).get();

    assert!(scalar.is_nan());
    assert!(wrapped.is_nan());
}

#[rstest]
#[case(
    vec![DistancePair::new(RowIndex::new(0), RowIndex::new(1))],
    vec![]
)]
#[case(
    vec![DistancePair::new(RowIndex::new(0), RowIndex::new(1))],
    vec![0.0, 1.0]
)]
fn batch_pairs_reject_mismatched_output_lengths(
    #[case] pairs: Vec<DistancePair>,
    #[case] mut out: Vec<f32>,
) {
    let matrix = RowMajorMatrix::new(
        MatrixValues::new(&[1.0, 2.0, 3.0, 4.0]),
        RowCount::new(2),
        Dimension::new(2),
    );
    let mut out_buffer = DistanceBuffer::new(&mut out);
    let err = euclidean_distance_batch_pairs(matrix, &pairs, &mut out_buffer)
        .expect_err("mismatched outputs must fail");
    assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
}

#[rstest]
fn batch_pairs_compute_distances(
    matrix_3x2: Result<RowMajorMatrix<'static>, DataSourceError>,
) -> Result<(), DataSourceError> {
    let matrix_3x2 = matrix_3x2?;
    let pairs = vec![
        DistancePair::new(RowIndex::new(0), RowIndex::new(1)),
        DistancePair::new(RowIndex::new(0), RowIndex::new(2)),
        DistancePair::new(RowIndex::new(2), RowIndex::new(1)),
    ];
    let mut out = vec![0.0_f32; pairs.len()];
    let mut out_buffer = DistanceBuffer::new(&mut out);

    euclidean_distance_batch_pairs(matrix_3x2, &pairs, &mut out_buffer)
        .expect("batch computation must succeed");

    close(Distance::new(out[0]), Distance::new(5.0_f32));
    close(Distance::new(out[1]), Distance::new((2.0_f32).sqrt()));
    close(Distance::new(out[2]), Distance::new((29.0_f32).sqrt()));
    Ok(())
}

#[rstest]
#[case(vec![f32::NAN, 0.0], vec![0.0, 0.0])]
#[case(vec![f32::INFINITY, 0.0], vec![0.0, 0.0])]
#[case(vec![f32::NEG_INFINITY, 0.0], vec![0.0, 0.0])]
#[case(vec![0.0, 0.0], vec![f32::NAN, 0.0])]
#[case(vec![0.0, 0.0], vec![f32::INFINITY, 0.0])]
#[case(vec![0.0, 0.0], vec![f32::NEG_INFINITY, 0.0])]
fn query_points_kernel_canonicalizes_non_finite_results_to_nan(
    #[case] query: Vec<f32>,
    #[case] point: Vec<f32>,
) -> Result<(), DataSourceError> {
    let mut values = query.clone();
    values.extend_from_slice(&point);
    let matrix = RowMajorMatrix::new(
        MatrixValues::new(&values),
        RowCount::new(2),
        Dimension::new(query.len()),
    );
    let query = matrix.row(RowIndex::new(0))?;
    let points = DensePointView::from_row_indices(matrix, &[RowIndex::new(1)])?;
    let mut scalar = vec![0.0_f32; 1];
    let mut actual = vec![0.0_f32; 1];

    kernels::euclidean_distance_query_points_scalar(query.as_slice(), &points, &mut scalar);
    kernels::euclidean_distance_query_points(query.as_slice(), &points, &mut actual);

    assert!(scalar[0].is_nan());
    assert!(actual[0].is_nan());
    Ok(())
}

#[rstest]
fn batch_pairs_leave_output_unmodified_on_error(
    matrix_3x2: Result<RowMajorMatrix<'static>, DataSourceError>,
) -> Result<(), DataSourceError> {
    let matrix_3x2 = matrix_3x2?;
    let pairs = vec![
        DistancePair::new(RowIndex::new(0), RowIndex::new(1)),
        DistancePair::new(RowIndex::new(0), RowIndex::new(9)),
    ];
    let mut out = vec![10.0_f32, 20.0_f32];
    let mut out_buffer = DistanceBuffer::new(&mut out);

    let err = euclidean_distance_batch_pairs(matrix_3x2, &pairs, &mut out_buffer)
        .expect_err("out-of-bounds pair must fail");

    assert!(matches!(err, DataSourceError::OutOfBounds { index: 9 }));
    assert_eq!(out, vec![10.0_f32, 20.0_f32]);
    Ok(())
}

#[rstest]
fn raw_pairs_preserve_original_validation_order_for_shared_query_batches(
    matrix_3x2: Result<RowMajorMatrix<'static>, DataSourceError>,
) -> Result<(), DataSourceError> {
    let matrix_3x2 = matrix_3x2?;
    let pairs = vec![(99, 1), (0, 1)];
    let mut out = vec![10.0_f32; pairs.len()];
    let mut out_buffer = DistanceBuffer::new(&mut out);

    let err = euclidean_distance_batch_raw_pairs(matrix_3x2, &pairs, &mut out_buffer)
        .expect_err("out-of-bounds pair must fail");

    assert_eq!(err, DataSourceError::OutOfBounds { index: 99 });
    assert_eq!(out, vec![10.0_f32; pairs.len()]);
    Ok(())
}

#[rstest]
fn query_points_kernel_matches_scalar_reference(
    matrix_3x2: Result<RowMajorMatrix<'static>, DataSourceError>,
) -> Result<(), DataSourceError> {
    let matrix_3x2 = matrix_3x2?;
    let query = matrix_3x2.row(RowIndex::new(0))?;
    let points =
        DensePointView::from_row_indices(matrix_3x2, &[RowIndex::new(1), RowIndex::new(2)])?;
    let mut out = vec![0.0_f32; 2];

    kernels::euclidean_distance_query_points(query.as_slice(), &points, &mut out);

    close(Distance::new(out[0]), Distance::new(5.0_f32));
    close(Distance::new(out[1]), Distance::new((2.0_f32).sqrt()));
    Ok(())
}
