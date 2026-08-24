//! Aligned Structure of Arrays packing for dense point batches.
//!
//! `DensePointView<'a>` repacks selected row-major points into a
//! dimension-major, lane-padded layout so batch kernels can read contiguous
//! coordinate blocks. The packed storage is 64-byte aligned and zero-padded to
//! a 16-lane multiple so both AVX2 and AVX-512 paths can safely read full
//! lanes.

use chutoro_core::DataSourceError;

use super::{Dimension, MAX_SIMD_LANES, RowIndex, RowMajorMatrix};

/// One 64-byte-aligned SIMD block.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
struct AlignedBlock([f32; MAX_SIMD_LANES]);

/// Owns lane-aligned blocks for a packed Structure-of-Arrays view.
#[derive(Debug)]
struct PackedSoaStorage {
    /// Consecutive aligned storage blocks.
    blocks: Vec<AlignedBlock>,
}

impl PackedSoaStorage {
    /// Allocate zero-filled storage for the requested scalar value count.
    fn zeroed(len: usize) -> Self {
        let blocks = len.div_ceil(MAX_SIMD_LANES);
        Self {
            blocks: vec![AlignedBlock([0.0; MAX_SIMD_LANES]); blocks],
        }
    }

    #[inline]
    /// Return the scalar capacity represented by the aligned blocks.
    const fn len(&self) -> usize {
        self.blocks.len() * MAX_SIMD_LANES
    }

    /// View the aligned blocks as contiguous scalar values.
    const fn as_slice(&self) -> &[f32] {
        let ptr = self.blocks.as_ptr().cast::<f32>();
        // Safety: `AlignedBlock` is `repr(C)` over `[f32; MAX_SIMD_LANES]`, so
        // the blocks are contiguous `f32` values with no interior padding.
        unsafe { std::slice::from_raw_parts(ptr, self.len()) }
    }

    /// View the aligned blocks as mutable contiguous scalar values.
    const fn as_mut_slice(&mut self) -> &mut [f32] {
        let ptr = self.blocks.as_mut_ptr().cast::<f32>();
        // Safety: `AlignedBlock` is `repr(C)` over `[f32; MAX_SIMD_LANES]`, so
        // the blocks are contiguous `f32` values with no interior padding.
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len()) }
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "packed storage is allocated for every dimension and padded point before block access"
    )]
    /// Return the packed coordinate block for one dimension.
    fn block(&self, dimension_index: usize, padded_point_count: usize) -> &[f32] {
        let start = dimension_index * padded_point_count;
        let end = start + padded_point_count;
        &self.as_slice()[start..end]
    }
}

/// Aligned Structure of Arrays packing for a selected dense point batch.
#[derive(Debug)]
pub(crate) struct DensePointView<'a> {
    /// Aligned Structure-of-Arrays storage for selected points.
    storage: PackedSoaStorage,
    /// Number of original points represented by the view.
    point_count: usize,
    /// Lane-padded number of points in every coordinate block.
    padded_point_count: usize,
    /// Number of coordinates in every packed point.
    dimension: Dimension,
    /// Binds the view lifetime to the source matrix values.
    _marker: std::marker::PhantomData<&'a [f32]>,
}

impl<'a> DensePointView<'a> {
    /// Packs the selected row indices into an aligned `SoA` layout.
    #[expect(
        clippy::indexing_slicing,
        reason = "packed storage reserves one padded coordinate block for every validated row dimension"
    )]
    pub(crate) fn from_row_indices(
        matrix: RowMajorMatrix<'a>,
        point_indices: &[RowIndex],
    ) -> Result<Self, DataSourceError> {
        let point_count = point_indices.len();
        let padded_point_count = padded_point_count(point_count);
        let dimension = matrix.dimension();
        let total_values = padded_point_count.saturating_mul(dimension.get());
        let mut storage = PackedSoaStorage::zeroed(total_values);
        let packed = storage.as_mut_slice();

        for (point_offset, index) in point_indices.iter().copied().enumerate() {
            let row = matrix.row(index)?;
            for (dimension_offset, value) in row.as_slice().iter().copied().enumerate() {
                packed[dimension_offset * padded_point_count + point_offset] = value;
            }
        }

        Ok(Self {
            storage,
            point_count,
            padded_point_count,
            dimension,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the number of logical points in the packed view.
    #[must_use]
    pub(crate) const fn point_count(&self) -> usize {
        self.point_count
    }

    /// Returns the zero-padded point count used for packed coordinate blocks.
    #[cfg(any(
        test,
        kani,
        all(
            feature = "simd_avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ),
        all(
            feature = "simd_avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ),
        all(
            feature = "simd_neon",
            any(target_arch = "arm", target_arch = "aarch64")
        ),
        all(feature = "nightly_portable_simd", nightly)
    ))]
    #[must_use]
    pub(crate) const fn padded_point_count(&self) -> usize {
        self.padded_point_count
    }

    /// Returns the number of scalar dimensions in each logical point.
    #[must_use]
    pub(crate) const fn dimension(&self) -> Dimension {
        self.dimension
    }

    /// Returns whether scalar fallback should be preferred for this view.
    #[must_use]
    pub(crate) const fn prefers_scalar_fallback(&self) -> bool {
        self.point_count <= 1 || self.dimension.get() == 0
    }

    /// Returns the packed values for a single coordinate across all points.
    #[must_use]
    pub(crate) fn coordinate_block(&self, dimension_index: usize) -> &[f32] {
        debug_assert!(
            dimension_index < self.dimension.get(),
            "coordinate block index must be within the packed dimension"
        );
        self.storage.block(dimension_index, self.padded_point_count)
    }

    /// Returns whether the packed storage base pointer satisfies `alignment`.
    #[must_use]
    pub(crate) fn is_aligned_to(&self, alignment: usize) -> bool {
        self.storage.as_slice().as_ptr().align_offset(alignment) == 0
    }
}

/// Round a point count up to the SIMD lane width.
pub(super) const fn padded_point_count(point_count: usize) -> usize {
    point_count.next_multiple_of(MAX_SIMD_LANES)
}
