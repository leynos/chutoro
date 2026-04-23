//! Compile-pass fixture verifying session API availability when the `cpu`
//! feature is enabled.

use std::sync::Arc;

use chutoro_core::{
    ChutoroBuilder, ClusteringSession, DataSource, DataSourceError, MetricDescriptor,
    SessionConfig, SessionRefreshPolicy,
};

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
    let session: ClusteringSession<Dummy> = ChutoroBuilder::new()
        .with_session_refresh_policy(SessionRefreshPolicy::manual())
        .build_session(source)
        .expect("cpu-enabled session API must compile");
    let config: &SessionConfig = session.config();
    let _ = config.refresh_policy();
}
