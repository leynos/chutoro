//! Shared fixtures and data sources for session tests.

use std::sync::Arc;

use rstest::fixture;

use crate::{ChutoroBuilder, ClusteringSession, DataSource, DataSourceError, MetricDescriptor};

#[derive(Clone, Debug)]
pub(super) struct SessionTestSource {
    values: Vec<f32>,
    name: &'static str,
}

impl SessionTestSource {
    pub(super) fn with_len(len: usize) -> Self {
        Self {
            values: (0..len).map(|value| value as f32).collect(),
            name: "session-test",
        }
    }
}

impl DataSource for SessionTestSource {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn name(&self) -> &str {
        self.name
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let left = self
            .values
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let right = self
            .values
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok((left - right).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("session-test:abs")
    }
}

#[fixture]
pub(super) fn session_builder() -> ChutoroBuilder {
    ChutoroBuilder::new()
}

pub(super) fn make_session(
    builder: ChutoroBuilder,
    source_len: usize,
) -> (ClusteringSession<SessionTestSource>, Arc<SessionTestSource>) {
    let source = Arc::new(SessionTestSource::with_len(source_len));
    let session = builder
        .build_session(Arc::clone(&source))
        .expect("session must build");
    (session, source)
}
