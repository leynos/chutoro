//! x86 and x86_64 SIMD kernel implementations.

use super::{DensePointView, finalize_distance, squared_l2_tail};

#[cfg(target_arch = "x86")]
use std::arch::x86 as x86_arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as x86_arch;

macro_rules! impl_squared_l2_x86_simd {
    (
        $fn_name:ident,
        cargo_feature = $cargo_feature:literal,
        target_feature = $target_feature:literal,
        lanes = $lanes:literal,
        zero = $zero:ident,
        load = $load:ident,
        sub = $sub:ident,
        mul = $mul:ident,
        add = $add:ident,
        store = $store:ident $(,)?
    ) => {
        #[cfg(all(feature = $cargo_feature, any(target_arch = "x86", target_arch = "x86_64")))]
        #[target_feature(enable = $target_feature)]
        pub(super) unsafe fn $fn_name(left: &[f32], right: &[f32]) -> f32 {
            let mut index = 0_usize;
            let mut acc = x86_arch::$zero();
            while index + $lanes <= left.len() {
                // Safety: `index + $lanes <= len` ensures in-bounds load.
                let left_chunk = unsafe { x86_arch::$load(left.as_ptr().add(index)) };
                // Safety: `index + $lanes <= len` ensures in-bounds load.
                let right_chunk = unsafe { x86_arch::$load(right.as_ptr().add(index)) };
                let delta = x86_arch::$sub(left_chunk, right_chunk);
                let squared = x86_arch::$mul(delta, delta);
                acc = x86_arch::$add(acc, squared);
                index += $lanes;
            }

            let mut lane_sum = [0.0_f32; $lanes];
            // Safety: `lane_sum` has exactly `$lanes` `f32` values.
            unsafe { x86_arch::$store(lane_sum.as_mut_ptr(), acc) };
            let mut total = lane_sum.iter().sum::<f32>();
            total += squared_l2_tail(left, right, index);
            total
        }
    };
}

macro_rules! impl_euclidean_distance_query_points_x86_simd {
    (
        $unsafe_fn:ident,
        $entry_fn:ident,
        cargo_feature = $cargo_feature:literal,
        target_feature = $target_feature:literal,
        lanes = $lanes:literal,
        setzero = $setzero:ident,
        set1 = $set1:ident,
        load = $load:ident,
        sub = $sub:ident,
        mul = $mul:ident,
        add = $add:ident,
        storeu = $storeu:ident $(,)?
    ) => {
        #[cfg(all(feature = $cargo_feature, any(target_arch = "x86", target_arch = "x86_64")))]
        pub(super) fn $entry_fn(query: &[f32], points: &DensePointView<'_>, out: &mut [f32]) {
            debug_assert_eq!(query.len(), points.dimension().get());
            debug_assert_eq!(out.len(), points.point_count());
            // Safety: this entrypoint is selected only after runtime feature
            // detection for the matching SIMD backend.
            unsafe { $unsafe_fn(query, points, out) }
        }

        #[cfg(all(feature = $cargo_feature, any(target_arch = "x86", target_arch = "x86_64")))]
        #[target_feature(enable = $target_feature)]
        unsafe fn $unsafe_fn(query: &[f32], points: &DensePointView<'_>, out: &mut [f32]) {
            let padded_count = points.padded_point_count();
            for offset in (0..padded_count).step_by($lanes) {
                let mut acc = x86_arch::$setzero();
                for (dimension_index, query_value) in query.iter().copied().enumerate() {
                    let query_lane = x86_arch::$set1(query_value);
                    let values = points.coordinate_block(dimension_index);
                    // Safety: `DensePointView` guarantees aligned blocks and
                    // lane-multiple padding, so full-lane loads are in bounds.
                    let point_lane = unsafe { x86_arch::$load(values.as_ptr().add(offset)) };
                    let delta = x86_arch::$sub(query_lane, point_lane);
                    acc = x86_arch::$add(acc, x86_arch::$mul(delta, delta));
                }
                let mut lane = [0.0_f32; $lanes];
                unsafe { x86_arch::$storeu(lane.as_mut_ptr(), acc) };
                let remaining = out.len().saturating_sub(offset).min($lanes);
                for lane_index in 0..remaining {
                    out[offset + lane_index] = finalize_distance(lane[lane_index].sqrt());
                }
            }
        }
    };
}

#[cfg(all(
    feature = "simd_avx512",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn euclidean_distance_avx512(left: &[f32], right: &[f32]) -> f32 {
    finalize_distance(unsafe { squared_l2_avx512(left, right) }.sqrt())
}

impl_squared_l2_x86_simd!(
    squared_l2_avx512,
    cargo_feature = "simd_avx512",
    target_feature = "avx512f",
    lanes = 16,
    zero = _mm512_setzero_ps,
    load = _mm512_loadu_ps,
    sub = _mm512_sub_ps,
    mul = _mm512_mul_ps,
    add = _mm512_add_ps,
    store = _mm512_storeu_ps,
);

impl_euclidean_distance_query_points_x86_simd!(
    euclidean_distance_query_points_avx512,
    euclidean_distance_query_points_avx512_entry,
    cargo_feature = "simd_avx512",
    target_feature = "avx512f",
    lanes = 16,
    setzero = _mm512_setzero_ps,
    set1 = _mm512_set1_ps,
    load = _mm512_load_ps,
    sub = _mm512_sub_ps,
    mul = _mm512_mul_ps,
    add = _mm512_add_ps,
    storeu = _mm512_storeu_ps,
);

#[cfg(all(
    feature = "simd_avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn euclidean_distance_avx2(left: &[f32], right: &[f32]) -> f32 {
    finalize_distance(unsafe { squared_l2_avx2(left, right) }.sqrt())
}

impl_squared_l2_x86_simd!(
    squared_l2_avx2,
    cargo_feature = "simd_avx2",
    target_feature = "avx2",
    lanes = 8,
    zero = _mm256_setzero_ps,
    load = _mm256_loadu_ps,
    sub = _mm256_sub_ps,
    mul = _mm256_mul_ps,
    add = _mm256_add_ps,
    store = _mm256_storeu_ps,
);

impl_euclidean_distance_query_points_x86_simd!(
    euclidean_distance_query_points_avx2,
    euclidean_distance_query_points_avx2_entry,
    cargo_feature = "simd_avx2",
    target_feature = "avx2",
    lanes = 8,
    setzero = _mm256_setzero_ps,
    set1 = _mm256_set1_ps,
    load = _mm256_load_ps,
    sub = _mm256_sub_ps,
    mul = _mm256_mul_ps,
    add = _mm256_add_ps,
    storeu = _mm256_storeu_ps,
);
