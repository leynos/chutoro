//! SIMD-aware Euclidean distance kernels for dense numeric providers.
//!
//! The implementation uses compile-time feature gates plus one-time runtime
//! dispatch to select the best available SIMD backend and falls back to a
//! scalar kernel otherwise.

use chutoro_core::{DataSourceError, PointPair};

mod dispatch;
#[cfg(kani)]
mod kani_proofs;
mod kernels;
mod point_view;
#[cfg(test)]
mod semantics;
mod types;

use dispatch::EuclideanBackend;
pub(crate) use point_view::DensePointView;
pub(crate) use types::{
    Dimension, Distance, DistanceBuffer, MatrixValues, RowCount, RowIndex, RowMajorMatrix, RowSlice,
};

/// Maximum scalar lanes handled by the supported SIMD backends.
pub(crate) const MAX_SIMD_LANES: usize = 16;
/// Byte alignment required by packed SIMD storage.
pub(crate) const SIMD_ALIGNMENT_BYTES: usize = 64;

/// Returns how many logical output lanes belong to a SIMD batch.
#[must_use]
pub(crate) fn lane_output_count(point_count: usize, offset: usize, lanes: usize) -> usize {
    point_count.saturating_sub(offset).min(lanes)
}

/// Computes Euclidean distance for two equal-length vectors.
#[must_use]
pub(crate) fn euclidean_distance(left: RowSlice<'_>, right: RowSlice<'_>) -> Distance {
    assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    let kernel = *kernels::EUCLIDEAN_KERNEL.get_or_init(kernels::select_euclidean_kernel);
    Distance::new(kernel(left.as_slice(), right.as_slice()))
}

/// Computes Euclidean distances for the ordered `PointPair`s in `pairs`.
///
/// The `values` slice must encode `rows * dimension` `f32` values in row-major
/// order. Results keep the order of `pairs`, and each pair is read as
/// `left`-to-`right`, so reversing a pair selects the mirrored rows.
pub(crate) fn euclidean_distance_batch_pairs(
    matrix: RowMajorMatrix<'_>,
    pairs: &[PointPair],
    out: &mut DistanceBuffer<'_>,
) -> Result<(), DataSourceError> {
    if pairs.len() != out.len() {
        return Err(DataSourceError::OutputLengthMismatch {
            out: out.capacity(),
            expected: pairs.len(),
        });
    }

    let results = collect_pair_results(matrix, pairs)?;

    for (value, slot) in results.into_iter().zip(out.slots_mut()) {
        *slot = value;
    }

    Ok(())
}

/// Compute every pair, packing a shared query when the backend benefits.
fn collect_pair_results(
    matrix: RowMajorMatrix<'_>,
    pairs: &[PointPair],
) -> Result<Vec<f32>, DataSourceError> {
    match shared_query_candidates(pairs) {
        Some((query, candidates))
            if should_pack_query_points(matrix.dimension().get(), candidates.len()) =>
        {
            collect_packed_query_distances(matrix, pairs, query, &candidates)
        }
        _ => collect_euclidean_distance_batch(
            matrix,
            pairs
                .iter()
                .map(|pair| (RowIndex::new(pair.left()), RowIndex::new(pair.right()))),
        ),
    }
}

/// Validate the pairs, then compute one query row against packed candidates.
fn collect_packed_query_distances(
    matrix: RowMajorMatrix<'_>,
    pairs: &[PointPair],
    query: RowIndex,
    candidates: &[RowIndex],
) -> Result<Vec<f32>, DataSourceError> {
    validate_pairs_in_order(matrix, pairs)?;
    let query_row = matrix.row(query)?;
    let point_view = DensePointView::from_row_indices(matrix, candidates)?;
    debug_assert!(!point_view.prefers_scalar_fallback());
    debug_assert!(point_view.is_aligned_to(SIMD_ALIGNMENT_BYTES));
    let mut results = vec![0.0_f32; candidates.len()];
    euclidean_distance_query_points(query_row, &point_view, &mut results)?;
    Ok(results)
}

/// Collect pairwise distances while preserving input pair order.
fn collect_euclidean_distance_batch(
    matrix: RowMajorMatrix<'_>,
    pairs: impl Iterator<Item = (RowIndex, RowIndex)>,
) -> Result<Vec<f32>, DataSourceError> {
    let (lower_bound, _) = pairs.size_hint();
    let mut results = Vec::with_capacity(lower_bound);
    for (left, right) in pairs {
        let left_row = matrix.row(left)?;
        let right_row = matrix.row(right)?;
        results.push(euclidean_distance(left_row, right_row).get());
    }
    Ok(results)
}

/// Compute distances from one query row to a packed point view.
fn euclidean_distance_query_points(
    query: RowSlice<'_>,
    points: &DensePointView<'_>,
    out: &mut [f32],
) -> Result<(), DataSourceError> {
    if out.len() != points.point_count() {
        return Err(DataSourceError::OutputLengthMismatch {
            out: out.len(),
            expected: points.point_count(),
        });
    }

    kernels::euclidean_distance_query_points(query.as_slice(), points, out);
    Ok(())
}

/// Decide whether the active backend benefits from packing query candidates.
fn should_pack_query_points(dimension: usize, candidate_count: usize) -> bool {
    should_pack_query_points_for_backend(dispatch::euclidean_backend(), dimension, candidate_count)
}

/// Decide whether one backend benefits from packing query candidates.
const fn should_pack_query_points_for_backend(
    backend: EuclideanBackend,
    dimension: usize,
    candidate_count: usize,
) -> bool {
    dimension > 0 && candidate_count > 1 && !matches!(backend, EuclideanBackend::Scalar)
}

/// Validate pair endpoints without changing their supplied order.
fn validate_pairs_in_order(
    matrix: RowMajorMatrix<'_>,
    pairs: &[PointPair],
) -> Result<(), DataSourceError> {
    let rows = matrix.rows().get();
    for pair in pairs {
        validate_raw_row_index(pair.left(), rows)?;
        validate_raw_row_index(pair.right(), rows)?;
    }
    Ok(())
}

/// Validate one raw row index against the matrix row count.
const fn validate_raw_row_index(index: usize, rows: usize) -> Result<(), DataSourceError> {
    if index < rows {
        Ok(())
    } else {
        Err(DataSourceError::OutOfBounds { index })
    }
}

/// Extract a common query and its candidate rows from homogeneous pairs.
fn shared_query_candidates(pairs: &[PointPair]) -> Option<(RowIndex, Vec<RowIndex>)> {
    let first = pairs.first().copied()?;
    if pairs.iter().all(|pair| pair.left() == first.left()) {
        return Some((
            RowIndex::new(first.left()),
            pairs
                .iter()
                .map(|pair| RowIndex::new(pair.right()))
                .collect(),
        ));
    }
    if pairs.iter().all(|pair| pair.right() == first.right()) {
        return Some((
            RowIndex::new(first.right()),
            pairs
                .iter()
                .map(|pair| RowIndex::new(pair.left()))
                .collect(),
        ));
    }
    None
}

#[cfg(test)]
mod tests;
