//! Compile-pass trybuild fixture verifying the public API surface of
//! [`ClusteringSession`] (defined in `chutoro-core/src/session/mod.rs`)
//! when the `cpu` Cargo feature is enabled.
//!
//! This file sits at the API-contract layer of the testing hierarchy: it is
//! compiled (but not executed) by the `trybuild` test harness to assert that
//! the expected symbols — including `build_session` and `append` — are
//! accessible and have the correct mutability requirements under the `cpu`
//! feature gate.  It does not make runtime assertions.

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
    let mut session: ClusteringSession<Dummy> = ChutoroBuilder::new()
        .with_session_refresh_policy(SessionRefreshPolicy::manual())
        .build_session(source)
        .expect("cpu-enabled session API must compile");
    session.append(&[0]).expect("append API must compile");
    let config: &SessionConfig = session.config();
    let _ = config.refresh_policy();
}
