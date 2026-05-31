//! Shared fixtures and data sources for session tests.
//!
//! This module provides the small in-memory [`SessionTestSource`], the
//! `session_builder` fixture, and helper constructors used by the focused
//! session test modules. Centralizing these fixtures keeps append, builder,
//! concurrency, metrics, and property tests aligned with production
//! [`super::ClusteringSession`] construction semantics.

use std::sync::Arc;

use rstest::fixture;

use crate::{
    CandidateEdge, ChutoroBuilder, ClusteringSession, CpuHnsw, DataSource, DataSourceError,
    HnswParams, MetricDescriptor,
};

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

pub(super) fn harvest_expected_edges(
    hnsw_params: HnswParams,
    source: &SessionTestSource,
    indices: &[usize],
) -> Vec<CandidateEdge> {
    let direct_index = CpuHnsw::with_capacity(hnsw_params, source.len().max(1))
        .expect("direct index must allocate");
    let mut expected_edges = Vec::new();
    for &index in indices {
        let edges = direct_index
            .insert_harvesting(index, source)
            .expect("direct insert must succeed");
        expected_edges.extend(edges);
    }
    expected_edges
}
