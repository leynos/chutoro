//! Shared fixtures and helpers for HNSW tests.

use crate::{DataSource, DataSourceError, hnsw::Neighbour};

#[derive(Clone)]
pub(super) struct DummySource {
    data: Vec<f32>,
}

impl DummySource {
    pub(super) fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl DataSource for DummySource {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &'static str {
        "dummy"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let a = self
            .data
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let b = self
            .data
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok(a.mul_add(1.0, std::ops::Neg::neg(*b)).abs())
    }
}

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
