//! Deterministic source shared by the CPU pipeline stage tests.

use crate::{DataSource, DataSourceError, MetricDescriptor};

/// Item count of the clustered fixture.
pub(super) const ITEM_COUNT: usize = 12;

/// Cluster count the fixture must produce.
pub(super) const EXPECTED_CLUSTERS: usize = 3;

/// Three well-separated groups of four points.
///
/// The spacing between groups is an order of magnitude larger than the spacing
/// within a group, so a dropped edge or a mis-weighted mutual-reachability edge
/// changes the resulting clustering instead of hiding inside within-group
/// noise.
const POINTS: [f32; ITEM_COUNT] = [
    0.0, 0.1, 0.2, 0.3, 5.0, 5.1, 5.2, 5.3, 10.0, 10.1, 10.2, 10.3,
];

/// A deterministic one-dimensional source with three well-separated groups.
pub(super) struct ClusteredSource {
    /// Scalar positions of the fixture items.
    points: Vec<f32>,
}

impl ClusteredSource {
    /// Creates the clustered fixture source.
    pub(super) fn new() -> Self {
        Self {
            points: POINTS.to_vec(),
        }
    }
}

impl DataSource for ClusteredSource {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn name(&self) -> &'static str {
        "clustered"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_value = self
            .points
            .get(left)
            .ok_or(DataSourceError::OutOfBounds { index: left })?;
        let right_value = self
            .points
            .get(right)
            .ok_or(DataSourceError::OutOfBounds { index: right })?;
        Ok(left_value
            .mul_add(1.0, std::ops::Neg::neg(*right_value))
            .abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("abs")
    }
}