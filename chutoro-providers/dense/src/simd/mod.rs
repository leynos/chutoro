//! SIMD-aware Euclidean distance kernels for dense numeric providers.
//!
//! The implementation uses compile-time feature gates plus one-time runtime
//! dispatch to select the best available SIMD backend and falls back to a
//! scalar kernel otherwise.

use chutoro_core::DataSourceError;

mod dispatch;
mod kernels;
mod point_view;
mod types;

use dispatch::EuclideanBackend;
pub(crate) use point_view::DensePointView;
pub(crate) use types::{
    Dimension, Distance, DistanceBuffer, MatrixValues, RowCount, RowIndex, RowMajorMatrix,
    RowSlice, row_slice,
};

#[cfg(test)]
pub(crate) use types::DistancePair;

pub(crate) const MAX_SIMD_LANES: usize = 16;
pub(crate) const SIMD_ALIGNMENT_BYTES: usize = 64;

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

/// Computes Euclidean distances for `(left, right)` row pairs.
///
/// The `values` slice must encode `rows * dimension` `f32` values in row-major
/// order.
#[cfg(test)]
pub(crate) fn euclidean_distance_batch_pairs(
    matrix: RowMajorMatrix<'_>,
    pairs: &[DistancePair],
    out: &mut DistanceBuffer<'_>,
) -> Result<(), DataSourceError> {
    if pairs.len() != out.len() {
        return Err(DataSourceError::OutputLengthMismatch {
            out: out.capacity(),
            expected: pairs.len(),
        });
    }

    let results = collect_euclidean_distance_batch(
        matrix,
        pairs
            .iter()
            .copied()
            .map(|pair| (pair.left(), pair.right())),
    )?;

    for (value, slot) in results.into_iter().zip(out.slots_mut()) {
        *slot = value;
    }

    Ok(())
}

/// Computes Euclidean distances for raw `(left, right)` row index pairs.
///
/// The `values` slice must encode `rows * dimension` `f32` values in row-major
/// order.
pub(crate) fn euclidean_distance_batch_raw_pairs(
    matrix: RowMajorMatrix<'_>,
    pairs: &[(usize, usize)],
    out: &mut DistanceBuffer<'_>,
) -> Result<(), DataSourceError> {
    if pairs.len() != out.len() {
        return Err(DataSourceError::OutputLengthMismatch {
            out: out.capacity(),
            expected: pairs.len(),
        });
    }

    let results = match shared_query_candidates(pairs)
        .filter(|(_, candidates)| should_pack_query_points(candidates.len()))
    {
        Some((query, candidates)) => {
            validate_raw_pairs_in_order(matrix, pairs)?;
            let query_row = row_slice(matrix, query)?;
            let point_view = DensePointView::from_row_indices(matrix, &candidates)?;
            debug_assert!(!point_view.prefers_scalar_fallback());
            debug_assert!(point_view.is_aligned_to(SIMD_ALIGNMENT_BYTES));
            let mut results = vec![0.0_f32; candidates.len()];
            euclidean_distance_query_points(query_row, &point_view, &mut results)?;
            results
        }
        None => collect_euclidean_distance_batch_from_raw_pairs(matrix, pairs)?,
    };

    for (value, slot) in results.into_iter().zip(out.slots_mut()) {
        *slot = value;
    }

    Ok(())
}

fn collect_euclidean_distance_batch_from_raw_pairs(
    matrix: RowMajorMatrix<'_>,
    pairs: &[(usize, usize)],
) -> Result<Vec<f32>, DataSourceError> {
    collect_euclidean_distance_batch(
        matrix,
        pairs
            .iter()
            .copied()
            .map(|(left, right)| (RowIndex::new(left), RowIndex::new(right))),
    )
}

fn collect_euclidean_distance_batch(
    matrix: RowMajorMatrix<'_>,
    pairs: impl Iterator<Item = (RowIndex, RowIndex)>,
) -> Result<Vec<f32>, DataSourceError> {
    let pairs = pairs;
    let (lower_bound, _) = pairs.size_hint();
    let mut results = Vec::with_capacity(lower_bound);
    for (left, right) in pairs {
        let left_row = row_slice(matrix, left)?;
        let right_row = row_slice(matrix, right)?;
        results.push(euclidean_distance(left_row, right_row).get());
    }
    Ok(results)
}

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

fn should_pack_query_points(candidate_count: usize) -> bool {
    should_pack_query_points_for_backend(dispatch::euclidean_backend(), candidate_count)
}

fn should_pack_query_points_for_backend(backend: EuclideanBackend, candidate_count: usize) -> bool {
    candidate_count > 1 && !matches!(backend, EuclideanBackend::Scalar)
}

fn validate_raw_pairs_in_order(
    matrix: RowMajorMatrix<'_>,
    pairs: &[(usize, usize)],
) -> Result<(), DataSourceError> {
    let rows = matrix.rows().get();
    for (left, right) in pairs.iter().copied() {
        validate_raw_row_index(left, rows)?;
        validate_raw_row_index(right, rows)?;
    }
    Ok(())
}

fn validate_raw_row_index(index: usize, rows: usize) -> Result<(), DataSourceError> {
    if index < rows {
        Ok(())
    } else {
        Err(DataSourceError::OutOfBounds { index })
    }
}

fn shared_query_candidates(pairs: &[(usize, usize)]) -> Option<(RowIndex, Vec<RowIndex>)> {
    let (first_left, first_right) = pairs.first().copied()?;
    if pairs.iter().all(|(left, _)| *left == first_left) {
        return Some((
            RowIndex::new(first_left),
            pairs
                .iter()
                .map(|(_, right)| RowIndex::new(*right))
                .collect(),
        ));
    }
    if pairs.iter().all(|(_, right)| *right == first_right) {
        return Some((
            RowIndex::new(first_right),
            pairs.iter().map(|(left, _)| RowIndex::new(*left)).collect(),
        ));
    }
    None
}

#[cfg(test)]
mod tests;
