//! Shared fixtures and data sources for session tests.
//!
//! This module provides the session-specific constructor for the shared
//! numeric source, the `session_builder` fixture, and helper constructors
//! used by focused session tests. Centralizing these fixtures keeps append,
//! builder, concurrency, metrics, and property tests aligned with production
//! [`super::ClusteringSession`] construction semantics.

use std::sync::Arc;

use rstest::fixture;

use crate::{
    CandidateEdge, ChutoroBuilder, ChutoroError, ClusteringSession, CpuHnsw, DataSource, HnswError,
    HnswParams, MetricDescriptor, test_utils::NumericTestSource,
};

/// Returns a fresh [`ChutoroBuilder`] for tests to customize.
#[fixture]
pub(super) fn session_builder() -> ChutoroBuilder {
    ChutoroBuilder::new()
}

/// Pairs a constructed [`ClusteringSession`] with the `Arc` handle to its
/// backing [`NumericTestSource`], so callers can inspect the source directly.
pub(super) type SessionAndSource = (ClusteringSession<NumericTestSource>, Arc<NumericTestSource>);

/// Builds the standard numeric source used by the session test suite.
///
/// # Examples
/// ```ignore
/// let source = session_test_source_with_len(3);
/// assert_eq!(source.name(), "session-test");
/// assert_eq!(source.metric_descriptor().as_str(), "session-test:abs");
/// ```
pub(super) fn session_test_source_with_len(len: usize) -> NumericTestSource {
    NumericTestSource::with_len("session-test", len)
        .using_metric_descriptor(MetricDescriptor::new("session-test:abs"))
}

/// Builds a [`NumericTestSource`] of `source_len` points and constructs a
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
    let source = Arc::new(session_test_source_with_len(source_len));
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
    source: &NumericTestSource,
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
