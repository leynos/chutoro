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
    CandidateEdge, ChutoroBuilder, ChutoroError, ClusteringSession, CpuHnsw, DataSource,
    DataSourceError, HnswError, HnswParams, MetricDescriptor,
};

/// An in-memory [`DataSource`] test double whose distance is the absolute
/// difference between two values.
///
/// The source always reports the fixed name `"session-test"`.
#[derive(Clone, Debug)]
pub(super) struct SessionTestSource {
    values: Vec<f32>,
    name: &'static str,
}

impl SessionTestSource {
    /// Builds a source of `len` points valued `0.0, 1.0, ..., (len - 1) as f32`.
    pub(super) fn with_len(len: usize) -> Self {
        Self {
            values: (0..len)
                .map(|value| value.to_string().parse::<f32>().unwrap_or(f32::INFINITY))
                .collect(),
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
        Ok(left.mul_add(1.0, std::ops::Neg::neg(*right)).abs())
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("session-test:abs")
    }
}

/// Returns a fresh [`ChutoroBuilder`] for tests to customize.
#[fixture]
pub(super) fn session_builder() -> ChutoroBuilder {
    ChutoroBuilder::new()
}

/// Pairs a constructed [`ClusteringSession`] with the `Arc` handle to its
/// backing [`SessionTestSource`], so callers can inspect the source directly.
pub(super) type SessionAndSource = (ClusteringSession<SessionTestSource>, Arc<SessionTestSource>);

/// Builds a [`SessionTestSource`] of `source_len` points and constructs a
/// session from it via `builder`.
///
/// # Errors
///
/// Propagates any [`ChutoroError`] returned by
/// [`ChutoroBuilder::build_session`], for example an invalid builder
/// configuration.
pub(super) fn make_session(
    builder: ChutoroBuilder,
    source_len: usize,
) -> Result<SessionAndSource, ChutoroError> {
    let source = Arc::new(SessionTestSource::with_len(source_len));
    let session = builder.build_session(Arc::clone(&source))?;
    Ok((session, source))
}

/// Builds an independent batch HNSW oracle and harvests candidate edges for
/// each of `indices`, in order, into a single accumulated `Vec`.
///
/// Used as ground truth against session-driven incremental insertion.
///
/// # Errors
///
/// Propagates any [`HnswError`] from [`CpuHnsw::with_capacity`] or from an
/// `insert_harvesting` call, short-circuiting on the first failure.
pub(super) fn harvest_expected_edges(
    hnsw_params: HnswParams,
    source: &SessionTestSource,
    indices: &[usize],
) -> Result<Vec<CandidateEdge>, HnswError> {
    let direct_index = CpuHnsw::with_capacity(hnsw_params, source.len().max(1))?;
    let mut expected_edges = Vec::new();
    for &index in indices {
        let edges = direct_index.insert_harvesting(index, source)?;
        expected_edges.extend(edges);
    }
    Ok(expected_edges)
}
