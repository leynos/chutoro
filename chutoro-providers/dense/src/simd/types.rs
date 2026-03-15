//! Shared typed wrappers and row access helpers for dense SIMD kernels.

use chutoro_core::DataSourceError;

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

    /// Returns the number of rows in the matrix.
    #[must_use]
    pub(crate) fn rows(self) -> RowCount {
        self.rows
    }

    /// Returns the matrix backing storage.
    #[must_use]
    pub(crate) fn values(self) -> MatrixValues<'a> {
        self.values
    }
}

/// Returns the row slice for a typed row index.
pub(crate) fn row_slice(
    matrix: RowMajorMatrix<'_>,
    index: RowIndex,
) -> Result<RowSlice<'_>, DataSourceError> {
    let raw_index = index.get();
    let raw_rows = matrix.rows().get();
    let raw_dimension = matrix.dimension().get();
    if raw_index >= raw_rows {
        return Err(DataSourceError::OutOfBounds { index: raw_index });
    }

    let start = raw_index
        .checked_mul(raw_dimension)
        .ok_or(DataSourceError::OutOfBounds { index: raw_index })?;
    let end = start
        .checked_add(raw_dimension)
        .ok_or(DataSourceError::OutOfBounds { index: raw_index })?;
    if end > matrix.values().len() {
        return Err(DataSourceError::OutOfBounds { index: raw_index });
    }

    let values = matrix.values().as_slice();
    Ok(RowSlice::new(&values[start..end]))
}
