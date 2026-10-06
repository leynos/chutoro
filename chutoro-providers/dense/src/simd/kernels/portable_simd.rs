//! Nightly-only portable SIMD Euclidean distance kernels.

use std::simd::Simd;

use super::{DensePointView, finalize_distance, squared_l2_tail};

/// Match the packed point view's 16-value alignment and padding unit.
const PORTABLE_SIMD_LANES: usize = 16;
/// Hold one complete packed point-view block; for example, 16 distances.
type PortableF32x16 = Simd<f32, PORTABLE_SIMD_LANES>;

/// Check matching row dimensions before entering the portable kernel.
/// Identical rows, for example, produce a zero distance.
pub(super) fn euclidean_distance_portable_simd_entry(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(
        left.len(),
        right.len(),
        "distance rows must have matching dimensions",
    );
    euclidean_distance_portable_simd(left, right)
}

/// Compute the Euclidean distance of two rows using portable SIMD.
/// Two identical rows, for example, have a distance of zero.
pub(super) fn euclidean_distance_portable_simd(left: &[f32], right: &[f32]) -> f32 {
    finalize_distance(squared_l2_portable_simd(left, right).sqrt())
}

/// Compute distances from a query to packed points in 16-lane blocks.
/// A packed point equal to the query, for example, yields zero in `out`.
pub(super) fn euclidean_distance_query_points_portable_simd_entry(
    query: &[f32],
    points: &DensePointView<'_>,
    out: &mut [f32],
) {
    debug_assert_eq!(query.len(), points.dimension().get());
    debug_assert_eq!(out.len(), points.point_count());

    for (chunk_index, outputs) in out.chunks_mut(PORTABLE_SIMD_LANES).enumerate() {
        let offset = chunk_index * PORTABLE_SIMD_LANES;
        let mut acc = PortableF32x16::splat(0.0);
        for (dimension_index, query_value) in query.iter().copied().enumerate() {
            let query_lane = PortableF32x16::splat(query_value);
            let values = points.coordinate_block(dimension_index);
            let Some(values_lane) = values.get(offset..offset + PORTABLE_SIMD_LANES) else {
                debug_assert!(
                    false,
                    "packed point block must contain every padded SIMD lane"
                );
                return;
            };
            let point_lane = PortableF32x16::from_slice(values_lane);
            let delta = query_lane - point_lane;
            acc += delta * delta;
        }

        for (output, value) in outputs.iter_mut().zip(acc.to_array()) {
            *output = finalize_distance(value.sqrt());
        }
    }
}

/// Sum squared differences for complete SIMD blocks and their scalar tail.
/// For a 17-value row, for example, the final value uses the scalar tail.
#[expect(
    clippy::float_arithmetic,
    reason = "squared-L2 accumulation combines genuine floating-point SIMD and scalar sums"
)]
fn squared_l2_portable_simd(left: &[f32], right: &[f32]) -> f32 {
    let mut index = 0_usize;
    let mut acc = PortableF32x16::splat(0.0);

    for (left_chunk, right_chunk) in left
        .chunks_exact(PORTABLE_SIMD_LANES)
        .zip(right.chunks_exact(PORTABLE_SIMD_LANES))
    {
        let left_lane = PortableF32x16::from_slice(left_chunk);
        let right_lane = PortableF32x16::from_slice(right_chunk);
        let delta = left_lane - right_lane;
        acc += delta * delta;
        index += PORTABLE_SIMD_LANES;
    }

    acc.to_array().into_iter().sum::<f32>() + squared_l2_tail(left, right, index)
}
