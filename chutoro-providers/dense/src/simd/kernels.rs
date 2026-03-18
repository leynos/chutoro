//! Internal SIMD and scalar squared-L2 kernels for Euclidean distance.
//!
//! # Note: Primitive Obsession suppression
//!
//! This module is excluded from the CodeScene Primitive Obsession rule
//! (see `.codescene/code-health-rules.json`). The kernel functions here
//! operate directly on `&[f32]` slices and `usize` offsets because SIMD
//! intrinsics require contiguous, unboxed memory and raw index arithmetic.
//! Introducing domain-type wrappers at this layer would add indirection on
//! every hot-path invocation, negating the benefit of SIMD acceleration.
//! Domain-typed wrappers (`RowSlice`, `RowIndex`, `Distance`, etc.) are
//! enforced at the public API boundary in `simd/mod.rs`; this private
//! module is intentionally exempted.

use std::sync::OnceLock;

use super::{DensePointView, dispatch};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(feature = "simd_avx2", feature = "simd_avx512")
))]
mod x86_simd;

#[cfg(all(
    feature = "simd_neon",
    any(target_arch = "arm", target_arch = "aarch64")
))]
mod neon_simd;

type EuclideanKernel = fn(&[f32], &[f32]) -> f32;
type EuclideanQueryPointsKernel = fn(&[f32], &DensePointView<'_>, &mut [f32]);

pub(super) static EUCLIDEAN_KERNEL: OnceLock<EuclideanKernel> = OnceLock::new();
static EUCLIDEAN_QUERY_POINTS_KERNEL: OnceLock<EuclideanQueryPointsKernel> = OnceLock::new();

macro_rules! select_backend_fn {
    (
        avx512 = $avx512:expr,
        avx2 = $avx2:expr,
        neon = $neon:expr,
        scalar = $scalar:expr $(,)?
    ) => {
        match dispatch::euclidean_backend() {
            #[cfg(all(
                feature = "simd_avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            dispatch::EuclideanBackend::Avx512 => $avx512,
            #[cfg(all(
                feature = "simd_avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            dispatch::EuclideanBackend::Avx2 => $avx2,
            #[cfg(all(
                feature = "simd_neon",
                any(target_arch = "arm", target_arch = "aarch64")
            ))]
            dispatch::EuclideanBackend::Neon => $neon,
            _ => $scalar,
        }
    };
}

pub(super) fn select_euclidean_kernel() -> EuclideanKernel {
    select_backend_fn!(
        avx512 = euclidean_distance_avx512_entry,
        avx2 = euclidean_distance_avx2_entry,
        neon = euclidean_distance_neon_entry,
        scalar = euclidean_distance_scalar,
    )
}

pub(super) fn euclidean_distance_scalar(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    finalize_distance(squared_l2_tail(left, right, 0).sqrt())
}

#[cfg(all(
    feature = "simd_avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub(super) fn euclidean_distance_avx2_entry(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    // Safety: this entrypoint is selected only after runtime AVX2 detection.
    unsafe { x86_simd::euclidean_distance_avx2(left, right) }
}

#[cfg(all(
    feature = "simd_avx512",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub(super) fn euclidean_distance_avx512_entry(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    // Safety: this entrypoint is selected only after runtime AVX-512F detection.
    unsafe { x86_simd::euclidean_distance_avx512(left, right) }
}

#[cfg(all(
    feature = "simd_neon",
    any(target_arch = "arm", target_arch = "aarch64")
))]
pub(super) fn euclidean_distance_neon_entry(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    // Safety: this entrypoint is selected only after runtime NEON detection on
    // `arm` or baseline availability on `aarch64`.
    unsafe { neon_simd::euclidean_distance_neon(left, right) }
}

pub(super) fn euclidean_distance_query_points(
    query: &[f32],
    points: &DensePointView<'_>,
    out: &mut [f32],
) {
    assert_eq!(query.len(), points.dimension().get());
    assert_eq!(out.len(), points.point_count());
    let kernel = *EUCLIDEAN_QUERY_POINTS_KERNEL.get_or_init(select_euclidean_query_points_kernel);
    kernel(query, points, out);
}

pub(super) fn euclidean_distance_query_points_scalar(
    query: &[f32],
    points: &DensePointView<'_>,
    out: &mut [f32],
) {
    debug_assert_eq!(query.len(), points.dimension().get());
    debug_assert_eq!(out.len(), points.point_count());
    out.fill(0.0);
    for (dimension_index, query_value) in query.iter().copied().enumerate() {
        for (distance, point_value) in out
            .iter_mut()
            .zip(points.coordinate_block(dimension_index).iter().copied())
        {
            let delta = query_value - point_value;
            *distance += delta * delta;
        }
    }
    for value in out.iter_mut() {
        *value = finalize_distance(value.sqrt());
    }
}

fn select_euclidean_query_points_kernel() -> EuclideanQueryPointsKernel {
    select_backend_fn!(
        avx512 = x86_simd::euclidean_distance_query_points_avx512_entry,
        avx2 = x86_simd::euclidean_distance_query_points_avx2_entry,
        neon = neon_simd::euclidean_distance_query_points_neon_entry,
        scalar = euclidean_distance_query_points_scalar,
    )
}

fn finalize_distance(value: f32) -> f32 {
    if value.is_finite() { value } else { f32::NAN }
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
