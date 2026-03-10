//! SIMD-aware Euclidean distance kernels for dense numeric providers.
//!
//! The implementation uses runtime dispatch on x86/x86_64 to select AVX-512 or
//! AVX2 specializations when available, and falls back to a scalar kernel
//! otherwise.

use chutoro_core::DataSourceError;

mod kernels;
mod point_view;

pub(crate) use point_view::DensePointView;

pub(crate) const MAX_SIMD_LANES: usize = 16;
pub(crate) const SIMD_ALIGNMENT_BYTES: usize = 64;

/// Logical row index into a row-major matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RowIndex(usize);

impl RowIndex {
    /// Builds a row index wrapper.
    #[must_use]
    pub(crate) fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the raw zero-based row index.
    #[must_use]
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

/// Pair of row indices used for batch distance computation.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DistancePair {
    left: RowIndex,
    right: RowIndex,
}

#[cfg(test)]
impl DistancePair {
    /// Builds a distance pair from typed row indices.
    #[must_use]
    pub(crate) fn new(left: RowIndex, right: RowIndex) -> Self {
        Self { left, right }
    }

    /// Returns the left row index.
    #[must_use]
    pub(crate) fn left(self) -> RowIndex {
        self.left
    }

    /// Returns the right row index.
    #[must_use]
    pub(crate) fn right(self) -> RowIndex {
        self.right
    }
}

/// Scalar distance value wrapper for domain-level intent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Distance(f32);

impl Distance {
    /// Builds a distance wrapper.
    #[must_use]
    pub(crate) fn new(value: f32) -> Self {
        Self(value)
    }

    /// Returns the raw distance value.
    #[must_use]
    pub(crate) fn get(self) -> f32 {
        self.0
    }
}

/// Matrix row width in scalar elements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Dimension(usize);

impl Dimension {
    /// Builds a dimension wrapper.
    #[must_use]
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the raw dimension value.
    #[must_use]
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

/// Number of rows in a row-major matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RowCount(usize);

impl RowCount {
    /// Builds a row count wrapper.
    #[must_use]
    pub(crate) fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the raw row count.
    #[must_use]
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

/// Mutable buffer for batch distance computation results.
pub(crate) struct DistanceBuffer<'a>(&'a mut [f32]);

impl<'a> DistanceBuffer<'a> {
    /// Builds a mutable distance output buffer wrapper.
    pub(crate) fn new(buffer: &'a mut [f32]) -> Self {
        Self(buffer)
    }

    /// Returns the number of writable distances.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the total buffer capacity in distance elements.
    #[must_use]
    pub(crate) fn capacity(&self) -> usize {
        self.0.len()
    }

    /// Returns mutable output slots for batch distance results.
    pub(crate) fn slots_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.0.iter_mut()
    }
}

/// Immutable view of a single row's scalar values.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowSlice<'a>(&'a [f32]);

impl<'a> RowSlice<'a> {
    /// Builds a row slice wrapper.
    #[must_use]
    pub(crate) fn new(slice: &'a [f32]) -> Self {
        Self(slice)
    }

    /// Returns the raw scalar slice.
    #[must_use]
    pub(crate) fn as_slice(self) -> &'a [f32] {
        self.0
    }

    /// Returns the number of scalar elements in the row.
    #[must_use]
    pub(crate) fn len(self) -> usize {
        self.0.len()
    }
}

/// Flat backing store for a row-major matrix.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixValues<'a>(&'a [f32]);

impl<'a> MatrixValues<'a> {
    /// Builds a matrix backing storage wrapper.
    #[must_use]
    pub(crate) fn new(values: &'a [f32]) -> Self {
        Self(values)
    }

    /// Returns the raw matrix values slice.
    #[must_use]
    pub(crate) fn as_slice(self) -> &'a [f32] {
        self.0
    }

    /// Returns the number of scalar values in the matrix backing store.
    #[must_use]
    pub(crate) fn len(self) -> usize {
        self.0.len()
    }
}

/// Row-major matrix metadata and storage for dense SIMD kernels.
#[derive(Clone, Copy)]
pub(crate) struct RowMajorMatrix<'a> {
    values: MatrixValues<'a>,
    rows: RowCount,
    dimension: Dimension,
}

impl<'a> RowMajorMatrix<'a> {
    /// Builds a row-major matrix view.
    #[must_use]
    pub(crate) fn new(values: MatrixValues<'a>, rows: RowCount, dimension: Dimension) -> Self {
        Self {
            values,
            rows,
            dimension,
        }
    }

    /// Returns the number of scalar dimensions in each row.
    #[must_use]
    pub(crate) fn dimension(self) -> Dimension {
        self.dimension
    }
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
    should_pack_query_points_for_backend(kernels::euclidean_backend(), candidate_count)
}

fn should_pack_query_points_for_backend(
    backend: kernels::EuclideanBackend,
    candidate_count: usize,
) -> bool {
    candidate_count > 1 && !matches!(backend, kernels::EuclideanBackend::Scalar)
}

fn validate_raw_pairs_in_order(
    matrix: RowMajorMatrix<'_>,
    pairs: &[(usize, usize)],
) -> Result<(), DataSourceError> {
    let rows = matrix.rows.get();
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

pub(crate) fn row_slice(
    matrix: RowMajorMatrix<'_>,
    index: RowIndex,
) -> Result<RowSlice<'_>, DataSourceError> {
    let raw_index = index.get();
    let raw_rows = matrix.rows.get();
    let raw_dimension = matrix.dimension.get();
    if raw_index >= raw_rows {
        return Err(DataSourceError::OutOfBounds { index: raw_index });
    }

    let start = raw_index
        .checked_mul(raw_dimension)
        .ok_or(DataSourceError::OutOfBounds { index: raw_index })?;
    let end = start
        .checked_add(raw_dimension)
        .ok_or(DataSourceError::OutOfBounds { index: raw_index })?;
    if end > matrix.values.len() {
        return Err(DataSourceError::OutOfBounds { index: raw_index });
    }

    let values = matrix.values.as_slice();
    Ok(RowSlice::new(&values[start..end]))
}

#[cfg(test)]
mod tests;
