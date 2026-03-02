//! SIMD-aware Euclidean distance kernels for dense numeric providers.
//!
//! The implementation uses runtime dispatch on x86/x86_64 to select AVX-512 or
//! AVX2 specializations when available, and falls back to a scalar kernel
//! otherwise.

use std::sync::OnceLock;

use chutoro_core::DataSourceError;

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

type EuclideanKernel = fn(&[f32], &[f32]) -> f32;

static EUCLIDEAN_KERNEL: OnceLock<EuclideanKernel> = OnceLock::new();

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

    /// Builds a distance pair from raw row indices.
    #[must_use]
    pub(crate) fn from_raw(left: usize, right: usize) -> Self {
        Self::new(RowIndex::new(left), RowIndex::new(right))
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

/// Row-major matrix metadata and storage for dense SIMD kernels.
#[derive(Clone, Copy)]
pub(crate) struct RowMajorMatrix<'a> {
    values: &'a [f32],
    rows: usize,
    dimension: usize,
}

impl<'a> RowMajorMatrix<'a> {
    /// Builds a row-major matrix view.
    #[must_use]
    pub(crate) fn new(values: &'a [f32], rows: usize, dimension: usize) -> Self {
        Self {
            values,
            rows,
            dimension,
        }
    }
}

/// Computes Euclidean distance for two equal-length vectors.
#[must_use]
pub(crate) fn euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    let kernel = *EUCLIDEAN_KERNEL.get_or_init(select_euclidean_kernel);
    kernel(left, right)
}

/// Computes Euclidean distances for `(left, right)` row pairs.
///
/// The `values` slice must encode `rows * dimension` `f32` values in row-major
/// order.
pub(crate) fn euclidean_distance_batch_pairs(
    matrix: RowMajorMatrix<'_>,
    pairs: &[DistancePair],
    out: &mut [f32],
) -> Result<(), DataSourceError> {
    if pairs.len() != out.len() {
        return Err(DataSourceError::OutputLengthMismatch {
            out: out.len(),
            expected: pairs.len(),
        });
    }

    for (pair, value) in pairs.iter().copied().zip(out.iter_mut()) {
        let left = pair.left().get();
        let right = pair.right().get();
        let left_row = row_slice(matrix, RowIndex::new(left))?;
        let right_row = row_slice(matrix, RowIndex::new(right))?;
        *value = euclidean_distance(left_row, right_row);
    }

    Ok(())
}

fn row_slice(matrix: RowMajorMatrix<'_>, index: RowIndex) -> Result<&[f32], DataSourceError> {
    let raw_index = index.get();
    if raw_index >= matrix.rows {
        return Err(DataSourceError::OutOfBounds { index: raw_index });
    }

    let start = raw_index
        .checked_mul(matrix.dimension)
        .ok_or(DataSourceError::OutOfBounds { index: raw_index })?;
    let end = start
        .checked_add(matrix.dimension)
        .ok_or(DataSourceError::OutOfBounds { index: raw_index })?;
    if end > matrix.values.len() {
        return Err(DataSourceError::OutOfBounds { index: raw_index });
    }

    Ok(&matrix.values[start..end])
}

fn select_euclidean_kernel() -> EuclideanKernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return euclidean_distance_avx512_entry;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return euclidean_distance_avx2_entry;
        }
    }

    euclidean_distance_scalar
}

fn euclidean_distance_scalar(left: &[f32], right: &[f32]) -> f32 {
    squared_l2_scalar(left, right).sqrt()
}

fn squared_l2_scalar(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| {
            let delta = *l - *r;
            delta * delta
        })
        .sum::<f32>()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn euclidean_distance_avx2_entry(left: &[f32], right: &[f32]) -> f32 {
    // Safety: this entrypoint is selected only after runtime AVX2 detection.
    unsafe { euclidean_distance_avx2(left, right) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn euclidean_distance_avx512_entry(left: &[f32], right: &[f32]) -> f32 {
    // Safety: this entrypoint is selected only after runtime AVX-512F detection.
    unsafe { euclidean_distance_avx512(left, right) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn euclidean_distance_avx512(left: &[f32], right: &[f32]) -> f32 {
    unsafe { squared_l2_avx512(left, right) }.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn squared_l2_avx512(left: &[f32], right: &[f32]) -> f32 {
    let mut index = 0_usize;
    let mut acc = arch::_mm512_setzero_ps();
    while index + 16 <= left.len() {
        // Safety: `index + 16 <= len` ensures the 16-lane load is in-bounds.
        let left_chunk = unsafe { arch::_mm512_loadu_ps(left.as_ptr().add(index)) };
        // Safety: `index + 16 <= len` ensures the 16-lane load is in-bounds.
        let right_chunk = unsafe { arch::_mm512_loadu_ps(right.as_ptr().add(index)) };
        let delta = arch::_mm512_sub_ps(left_chunk, right_chunk);
        let squared = arch::_mm512_mul_ps(delta, delta);
        acc = arch::_mm512_add_ps(acc, squared);
        index += 16;
    }

    let mut lane_sum = [0.0_f32; 16];
    // Safety: `lane_sum` has exactly 16 `f32` values.
    unsafe { arch::_mm512_storeu_ps(lane_sum.as_mut_ptr(), acc) };
    let mut total = lane_sum.iter().sum::<f32>();
    total += squared_l2_tail(left, right, index);
    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(left: &[f32], right: &[f32]) -> f32 {
    unsafe { squared_l2_avx2(left, right) }.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn squared_l2_avx2(left: &[f32], right: &[f32]) -> f32 {
    let mut index = 0_usize;
    let mut acc = arch::_mm256_setzero_ps();
    while index + 8 <= left.len() {
        // Safety: `index + 8 <= len` ensures the 8-lane load is in-bounds.
        let left_chunk = unsafe { arch::_mm256_loadu_ps(left.as_ptr().add(index)) };
        // Safety: `index + 8 <= len` ensures the 8-lane load is in-bounds.
        let right_chunk = unsafe { arch::_mm256_loadu_ps(right.as_ptr().add(index)) };
        let delta = arch::_mm256_sub_ps(left_chunk, right_chunk);
        let squared = arch::_mm256_mul_ps(delta, delta);
        acc = arch::_mm256_add_ps(acc, squared);
        index += 8;
    }

    let mut lane_sum = [0.0_f32; 8];
    // Safety: `lane_sum` has exactly 8 `f32` values.
    unsafe { arch::_mm256_storeu_ps(lane_sum.as_mut_ptr(), acc) };
    let mut total = lane_sum.iter().sum::<f32>();
    total += squared_l2_tail(left, right, index);
    total
}

fn squared_l2_tail(left: &[f32], right: &[f32], offset: usize) -> f32 {
    left[offset..]
        .iter()
        .zip(right[offset..].iter())
        .map(|(l, r)| {
            let delta = *l - *r;
            delta * delta
        })
        .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn close(left: f32, right: f32) {
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
        let expected = euclidean_distance_scalar(&left, &right);
        let actual = euclidean_distance(&left, &right);
        close(actual, expected);
    }

    #[rstest]
    #[case(vec![DistancePair::from_raw(0, 1)], vec![])]
    #[case(vec![DistancePair::from_raw(0, 1)], vec![0.0, 1.0])]
    fn batch_pairs_reject_mismatched_output_lengths(
        #[case] pairs: Vec<DistancePair>,
        #[case] mut out: Vec<f32>,
    ) {
        let matrix = RowMajorMatrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let err = euclidean_distance_batch_pairs(matrix, &pairs, &mut out)
            .expect_err("mismatched outputs must fail");
        assert!(matches!(err, DataSourceError::OutputLengthMismatch { .. }));
    }

    #[test]
    fn batch_pairs_compute_distances() {
        let values = vec![1.0, 2.0, 4.0, 6.0, 2.0, 1.0];
        let pairs = vec![
            DistancePair::from_raw(0, 1),
            DistancePair::from_raw(0, 2),
            DistancePair::from_raw(2, 1),
        ];
        let mut out = vec![0.0_f32; pairs.len()];
        let matrix = RowMajorMatrix::new(&values, 3, 2);

        euclidean_distance_batch_pairs(matrix, &pairs, &mut out)
            .expect("batch computation must succeed");

        close(out[0], 5.0_f32);
        close(out[1], (2.0_f32).sqrt());
        close(out[2], (29.0_f32).sqrt());
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

        let expected = euclidean_distance_scalar(&left, &right);
        let actual = euclidean_distance_avx2_entry(&left, &right);
        close(actual, expected);
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

        let expected = euclidean_distance_scalar(&left, &right);
        let actual = euclidean_distance_avx512_entry(&left, &right);
        close(actual, expected);
    }
}
