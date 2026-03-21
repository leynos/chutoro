//! ARM NEON SIMD kernel implementations.

use super::{DensePointView, finalize_distance, squared_l2_tail};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64 as arm_arch;
#[cfg(target_arch = "arm")]
use std::arch::arm as arm_arch;

#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
pub(super) unsafe fn euclidean_distance_neon(left: &[f32], right: &[f32]) -> f32 {
    finalize_distance(unsafe { squared_l2_neon(left, right) }.sqrt())
}

#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
unsafe fn squared_l2_neon(left: &[f32], right: &[f32]) -> f32 {
    let mut index = 0_usize;
    let mut acc = arm_arch::vdupq_n_f32(0.0);
    while index + 4 <= left.len() {
        // Safety: `index + 4 <= len` ensures in-bounds load.
        let left_chunk = unsafe { arm_arch::vld1q_f32(left.as_ptr().add(index)) };
        // Safety: `index + 4 <= len` ensures in-bounds load.
        let right_chunk = unsafe { arm_arch::vld1q_f32(right.as_ptr().add(index)) };
        let delta = arm_arch::vsubq_f32(left_chunk, right_chunk);
        let squared = arm_arch::vmulq_f32(delta, delta);
        acc = arm_arch::vaddq_f32(acc, squared);
        index += 4;
    }

    let mut lane_sum = [0.0_f32; 4];
    // Safety: `lane_sum` has exactly four `f32` values.
    unsafe { arm_arch::vst1q_f32(lane_sum.as_mut_ptr(), acc) };
    let mut total = lane_sum.iter().sum::<f32>();
    total += squared_l2_tail(left, right, index);
    total
}

pub(super) fn euclidean_distance_query_points_neon_entry(
    query: &[f32],
    points: &DensePointView<'_>,
    out: &mut [f32],
) {
    debug_assert_eq!(query.len(), points.dimension().get());
    debug_assert_eq!(out.len(), points.point_count());
    // Safety: this entrypoint is selected only after runtime NEON detection on
    // `arm` or baseline availability on `aarch64`.
    unsafe { euclidean_distance_query_points_neon(query, points, out) }
}

#[cfg_attr(target_arch = "arm", target_feature(enable = "neon"))]
unsafe fn euclidean_distance_query_points_neon(
    query: &[f32],
    points: &DensePointView<'_>,
    out: &mut [f32],
) {
    let padded_count = points.padded_point_count();
    for offset in (0..padded_count).step_by(4) {
        let mut acc = arm_arch::vdupq_n_f32(0.0);
        for (dimension_index, query_value) in query.iter().copied().enumerate() {
            let query_lane = arm_arch::vdupq_n_f32(query_value);
            let values = points.coordinate_block(dimension_index);
            // Safety: `DensePointView` guarantees aligned blocks and
            // lane-multiple padding, so full-lane loads are in bounds.
            let point_lane = unsafe { arm_arch::vld1q_f32(values.as_ptr().add(offset)) };
            let delta = arm_arch::vsubq_f32(query_lane, point_lane);
            acc = arm_arch::vaddq_f32(acc, arm_arch::vmulq_f32(delta, delta));
        }

        let mut lane = [0.0_f32; 4];
        // Safety: `lane` has exactly four `f32` values.
        unsafe { arm_arch::vst1q_f32(lane.as_mut_ptr(), acc) };
        let remaining = out.len().saturating_sub(offset).min(4);
        for lane_index in 0..remaining {
            out[offset + lane_index] = finalize_distance(lane[lane_index].sqrt());
        }
    }
}
