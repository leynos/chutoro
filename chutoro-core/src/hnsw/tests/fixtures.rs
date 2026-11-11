//! Shared fixtures and helpers for HNSW tests.

use crate::hnsw::Neighbour;

pub(super) fn assert_sorted_by_distance(neighbours: &[Neighbour]) {
    for window in neighbours.windows(2) {
        if let [left, right] = window {
            assert!(
                left.distance <= right.distance.mul_add(1.0, f32::EPSILON),
                "distances must be non-decreasing: {neighbours:?}",
            );
        }
    }
}
