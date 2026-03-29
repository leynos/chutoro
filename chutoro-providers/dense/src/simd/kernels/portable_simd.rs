//! Nightly-only portable SIMD Euclidean distance kernels.

use std::simd::Simd;

use super::{DensePointView, finalize_distance, squared_l2_tail};

const PORTABLE_SIMD_LANES: usize = 16;
type PortableF32x16 = Simd<f32, PORTABLE_SIMD_LANES>;

pub(super) fn euclidean_distance_portable_simd_entry(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    euclidean_distance_portable_simd(left, right)
}

pub(super) fn euclidean_distance_portable_simd(left: &[f32], right: &[f32]) -> f32 {
    finalize_distance(squared_l2_portable_simd(left, right).sqrt())
}

pub(super) fn euclidean_distance_query_points_portable_simd_entry(
    query: &[f32],
    points: &DensePointView<'_>,
    out: &mut [f32],
) {
    debug_assert_eq!(query.len(), points.dimension().get());
    debug_assert_eq!(out.len(), points.point_count());

    let padded_count = points.padded_point_count();
    for offset in (0..padded_count).step_by(PORTABLE_SIMD_LANES) {
        let mut acc = PortableF32x16::splat(0.0);
        for (dimension_index, query_value) in query.iter().copied().enumerate() {
            let query_lane = PortableF32x16::splat(query_value);
            let values = points.coordinate_block(dimension_index);
            let point_lane =
                PortableF32x16::from_slice(&values[offset..offset + PORTABLE_SIMD_LANES]);
            let delta = query_lane - point_lane;
            acc += delta * delta;
        }

        let remaining = out.len().saturating_sub(offset).min(PORTABLE_SIMD_LANES);
        for (lane_index, value) in acc.to_array().into_iter().take(remaining).enumerate() {
            out[offset + lane_index] = finalize_distance(value.sqrt());
        }
    }
}

fn squared_l2_portable_simd(left: &[f32], right: &[f32]) -> f32 {
    let mut index = 0_usize;
    let mut acc = PortableF32x16::splat(0.0);

    while index + PORTABLE_SIMD_LANES <= left.len() {
        let left_lane = PortableF32x16::from_slice(&left[index..index + PORTABLE_SIMD_LANES]);
        let right_lane = PortableF32x16::from_slice(&right[index..index + PORTABLE_SIMD_LANES]);
        let delta = left_lane - right_lane;
        acc += delta * delta;
        index += PORTABLE_SIMD_LANES;
    }

    acc.to_array().into_iter().sum::<f32>() + squared_l2_tail(left, right, index)
}
