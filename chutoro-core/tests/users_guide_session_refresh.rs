//! Runs the users' guide's `append -> refresh -> labels` example.
//!
//! The guide shows one complete program. This is an independent copy of that
//! snippet: it exercises the same API calls and asserts the same outcomes, but
//! it does not read the guide text, so editing the guide does not fail it.
//! Keep the two in step by hand.

#![cfg(feature = "cpu")]

use std::sync::Arc;

use chutoro_core::{ChutoroBuilder, DataSource, DataSourceError, MetricDescriptor};

/// The scalar source the guide's example defines.
struct Numbers(Vec<f32>);

impl DataSource for Numbers {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn name(&self) -> &'static str {
        "numbers"
    }

    #[expect(
        clippy::float_arithmetic,
        reason = "the guide's example is an absolute scalar distance"
    )]
    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let a = self
            .0
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let b = self
            .0
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((a - b).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("numbers:abs")
    }
}

#[test]
fn the_users_guide_refresh_example_runs() {
    // The body below is the users' guide's example, kept executable.
    let source = Arc::new(Numbers(vec![0.0, 1.0, 2.0, 3.0]));
    let mut session: chutoro_core::ClusteringSession<Numbers> = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .build_session(source)
        .expect("the guide's session must build");

    session.append(&[0, 1, 2, 3]).expect("append must succeed");
    session.refresh().expect("refresh must succeed");

    let labels = session.labels();
    assert_eq!(labels.len(), 4);
    assert_eq!(session.snapshot_version(), 1);
}
