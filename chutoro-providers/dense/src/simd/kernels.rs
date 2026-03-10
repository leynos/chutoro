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

use super::DensePointView;

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EuclideanBackend {
    Scalar,
    Avx2,
    Avx512,
}

type EuclideanKernel = fn(&[f32], &[f32]) -> f32;
type EuclideanQueryPointsKernel = fn(&[f32], &DensePointView<'_>, &mut [f32]);

pub(super) static EUCLIDEAN_KERNEL: OnceLock<EuclideanKernel> = OnceLock::new();
pub(super) static EUCLIDEAN_BACKEND: OnceLock<EuclideanBackend> = OnceLock::new();
static EUCLIDEAN_QUERY_POINTS_KERNEL: OnceLock<EuclideanQueryPointsKernel> = OnceLock::new();

pub(super) fn select_euclidean_kernel() -> EuclideanKernel {
    match select_euclidean_backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        EuclideanBackend::Avx512 => euclidean_distance_avx512_entry,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        EuclideanBackend::Avx2 => euclidean_distance_avx2_entry,
        _ => euclidean_distance_scalar,
    }
}

pub(super) fn select_euclidean_backend() -> EuclideanBackend {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return EuclideanBackend::Avx512;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return EuclideanBackend::Avx2;
        }
    }

    EuclideanBackend::Scalar
}

pub(super) fn euclidean_backend() -> EuclideanBackend {
    *EUCLIDEAN_BACKEND.get_or_init(select_euclidean_backend)
}

pub(super) fn euclidean_distance_scalar(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    squared_l2_tail(left, right, 0).sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) fn euclidean_distance_avx2_entry(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    // Safety: this entrypoint is selected only after runtime AVX2 detection.
    unsafe { euclidean_distance_avx2(left, right) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) fn euclidean_distance_avx512_entry(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    // Safety: this entrypoint is selected only after runtime AVX-512F detection.
    unsafe { euclidean_distance_avx512(left, right) }
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
        *value = value.sqrt();
    }
}

fn select_euclidean_query_points_kernel() -> EuclideanQueryPointsKernel {
    match euclidean_backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        EuclideanBackend::Avx512 => euclidean_distance_query_points_avx512_entry,
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        EuclideanBackend::Avx2 => euclidean_distance_query_points_avx2_entry,
        _ => euclidean_distance_query_points_scalar,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn euclidean_distance_avx512(left: &[f32], right: &[f32]) -> f32 {
    unsafe { squared_l2_avx512(left, right) }.sqrt()
}

/// Generates a squared-L2 SIMD kernel for a given target feature and lane width.
macro_rules! impl_squared_l2_simd {
    (
        $fn_name:ident,
        feature = $feature:literal,
        lanes = $lanes:literal,
        zero = $zero:ident,
        load = $load:ident,
        sub = $sub:ident,
        mul = $mul:ident,
        add = $add:ident,
        store = $store:ident $(,)?
    ) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = $feature)]
        unsafe fn $fn_name(left: &[f32], right: &[f32]) -> f32 {
            let mut index = 0_usize;
            let mut acc = arch::$zero();
            while index + $lanes <= left.len() {
                // Safety: `index + $lanes <= len` ensures in-bounds load.
                let left_chunk = unsafe { arch::$load(left.as_ptr().add(index)) };
                // Safety: `index + $lanes <= len` ensures in-bounds load.
                let right_chunk = unsafe { arch::$load(right.as_ptr().add(index)) };
                let delta = arch::$sub(left_chunk, right_chunk);
                let squared = arch::$mul(delta, delta);
                acc = arch::$add(acc, squared);
                index += $lanes;
            }

            let mut lane_sum = [0.0_f32; $lanes];
            // Safety: `lane_sum` has exactly `$lanes` `f32` values.
            unsafe { arch::$store(lane_sum.as_mut_ptr(), acc) };
            let mut total = lane_sum.iter().sum::<f32>();
            total += squared_l2_tail(left, right, index);
            total
        }
    };
}

macro_rules! impl_euclidean_distance_query_points_simd {
    (
        $unsafe_fn:ident,
        $entry_fn:ident,
        feature = $feature:literal,
        lanes = $lanes:literal,
        setzero = $setzero:ident,
        set1 = $set1:ident,
        load = $load:ident,
        sub = $sub:ident,
        mul = $mul:ident,
        add = $add:ident,
        storeu = $storeu:ident $(,)?
    ) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        fn $entry_fn(query: &[f32], points: &DensePointView<'_>, out: &mut [f32]) {
            debug_assert_eq!(query.len(), points.dimension().get());
            debug_assert_eq!(out.len(), points.point_count());
            // Safety: this entrypoint is selected only after runtime feature
            // detection for the matching SIMD backend.
            unsafe { $unsafe_fn(query, points, out) }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = $feature)]
        unsafe fn $unsafe_fn(query: &[f32], points: &DensePointView<'_>, out: &mut [f32]) {
            let padded_count = points.padded_point_count();
            for offset in (0..padded_count).step_by($lanes) {
                let mut acc = arch::$setzero();
                for (dimension_index, query_value) in query.iter().copied().enumerate() {
                    let query_lane = arch::$set1(query_value);
                    let values = points.coordinate_block(dimension_index);
                    // Safety: `DensePointView` guarantees aligned blocks and
                    // lane-multiple padding, so full-lane loads are in bounds.
                    let point_lane = unsafe { arch::$load(values.as_ptr().add(offset)) };
                    let delta = arch::$sub(query_lane, point_lane);
                    acc = arch::$add(acc, arch::$mul(delta, delta));
                }
                let mut lane = [0.0_f32; $lanes];
                unsafe { arch::$storeu(lane.as_mut_ptr(), acc) };
                let remaining = out.len().saturating_sub(offset).min($lanes);
                for lane_index in 0..remaining {
                    out[offset + lane_index] = lane[lane_index].sqrt();
                }
            }
        }
    };
}

impl_squared_l2_simd!(
    squared_l2_avx512,
    feature = "avx512f",
    lanes = 16,
    zero = _mm512_setzero_ps,
    load = _mm512_loadu_ps,
    sub = _mm512_sub_ps,
    mul = _mm512_mul_ps,
    add = _mm512_add_ps,
    store = _mm512_storeu_ps,
);

impl_euclidean_distance_query_points_simd!(
    euclidean_distance_query_points_avx512,
    euclidean_distance_query_points_avx512_entry,
    feature = "avx512f",
    lanes = 16,
    setzero = _mm512_setzero_ps,
    set1 = _mm512_set1_ps,
    load = _mm512_load_ps,
    sub = _mm512_sub_ps,
    mul = _mm512_mul_ps,
    add = _mm512_add_ps,
    storeu = _mm512_storeu_ps,
);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(left: &[f32], right: &[f32]) -> f32 {
    unsafe { squared_l2_avx2(left, right) }.sqrt()
}

impl_squared_l2_simd!(
    squared_l2_avx2,
    feature = "avx2",
    lanes = 8,
    zero = _mm256_setzero_ps,
    load = _mm256_loadu_ps,
    sub = _mm256_sub_ps,
    mul = _mm256_mul_ps,
    add = _mm256_add_ps,
    store = _mm256_storeu_ps,
);

impl_euclidean_distance_query_points_simd!(
    euclidean_distance_query_points_avx2,
    euclidean_distance_query_points_avx2_entry,
    feature = "avx2",
    lanes = 8,
    setzero = _mm256_setzero_ps,
    set1 = _mm256_set1_ps,
    load = _mm256_load_ps,
    sub = _mm256_sub_ps,
    mul = _mm256_mul_ps,
    add = _mm256_add_ps,
    storeu = _mm256_storeu_ps,
);

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
