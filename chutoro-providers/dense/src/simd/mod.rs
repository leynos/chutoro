//! SIMD-aware Euclidean distance kernels for dense numeric providers.
//!
//! The implementation uses runtime dispatch on x86/x86_64 to select AVX-512 or
//! AVX2 specializations when available, and falls back to a scalar kernel
//! otherwise.

use chutoro_core::DataSourceError;

mod kernels;

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DistancePair {
    left: RowIndex,
    right: RowIndex,
}

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

    for (pair, slot) in pairs.iter().copied().zip(out.slots_mut()) {
        let left_row = row_slice(matrix, pair.left())?;
        let right_row = row_slice(matrix, pair.right())?;
        *slot = euclidean_distance(left_row, right_row).get();
    }

    Ok(())
}

fn row_slice(matrix: RowMajorMatrix<'_>, index: RowIndex) -> Result<RowSlice<'_>, DataSourceError> {
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
mod tests {
    use super::kernels;
    use super::*;
    use rstest::rstest;

    fn close(left: Distance, right: Distance) {
        let left = left.get();
        let right = right.get();
        let tolerance = 1.0e-6_f32;
        assert!(
            (left - right).abs() <= tolerance,
            "left={left}, right={right}, tolerance={tolerance}",
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
    fn euclidean_distance_matches_scalar_reference(
        #[case] left: Vec<f32>,
        #[case] right: Vec<f32>,
    ) {
        let expected = kernels::euclidean_distance_scalar(&left, &right);
        let actual = euclidean_distance(RowSlice::new(&left), RowSlice::new(&right));
        close(actual, Distance::new(expected));
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

    #[test]
    fn batch_pairs_compute_distances() {
        let values = vec![1.0, 2.0, 4.0, 6.0, 2.0, 1.0];
        let pairs = vec![
            DistancePair::new(RowIndex::new(0), RowIndex::new(1)),
            DistancePair::new(RowIndex::new(0), RowIndex::new(2)),
            DistancePair::new(RowIndex::new(2), RowIndex::new(1)),
        ];
        let mut out = vec![0.0_f32; pairs.len()];
        let matrix = RowMajorMatrix::new(
            MatrixValues::new(&values),
            RowCount::new(3),
            Dimension::new(2),
        );
        let mut out_buffer = DistanceBuffer::new(&mut out);

        euclidean_distance_batch_pairs(matrix, &pairs, &mut out_buffer)
            .expect("batch computation must succeed");

        close(Distance::new(out[0]), Distance::new(5.0_f32));
        close(Distance::new(out[1]), Distance::new((2.0_f32).sqrt()));
        close(Distance::new(out[2]), Distance::new((29.0_f32).sqrt()));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_kernel_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        let left: Vec<f32> = (0_u32..35_u32).map(|index| index as f32 * 0.5).collect();
        let right: Vec<f32> = (0_u32..35_u32)
            .map(|index| (35_u32 - index) as f32 * 0.25)
            .collect();

        let expected = kernels::euclidean_distance_scalar(&left, &right);
        let actual = kernels::euclidean_distance_avx2_entry(&left, &right);
        close(Distance::new(actual), Distance::new(expected));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx512_entrypoint_matches_scalar_when_available() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }

        let left: Vec<f32> = (0_u32..67_u32).map(|index| index as f32 * 0.125).collect();
        let right: Vec<f32> = (0_u32..67_u32)
            .map(|index| (67_u32 - index) as f32 * 0.375)
            .collect();

        let expected = kernels::euclidean_distance_scalar(&left, &right);
        let actual = kernels::euclidean_distance_avx512_entry(&left, &right);
        close(Distance::new(actual), Distance::new(expected));
    }
}
