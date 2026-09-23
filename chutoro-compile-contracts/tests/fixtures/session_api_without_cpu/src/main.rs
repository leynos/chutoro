//! Compile-fail fixture verifying that session APIs stay unavailable without
//! the `cpu` feature.

use std::sync::Arc;

use chutoro_core::{ChutoroBuilder, DataSource, DataSourceError, MetricDescriptor};

struct Dummy(Vec<f32>);

impl DataSource for Dummy {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn name(&self) -> &str {
        "dummy"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let left = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((left - right).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("abs")
    }
}

fn main() {
    let source = Arc::new(Dummy(vec![0.0, 1.0]));
    let session: chutoro_core::ClusteringSession<Dummy> = ChutoroBuilder::new()
        .build_session(source)
        .expect("cpu-disabled session API should not compile");
    let config: &chutoro_core::SessionConfig = session.config();
    let _ = chutoro_core::SessionRefreshPolicy::manual();
    let _ = config.refresh_policy();
}
