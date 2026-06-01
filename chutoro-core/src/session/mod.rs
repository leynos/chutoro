//! CPU-backed session types for incremental clustering workflows.
//!
//! This module provides the public session surface used by the roadmap's
//! incremental clustering work. The session owns configuration, an allocated
//! HNSW index, and the backing data source. It currently supports append-only
//! insertion and buffers harvested candidate edges for later refresh work.

use std::sync::Arc;

use crate::{CandidateEdge, CpuHnsw, DataSource, DataSourceError, MetricDescriptor, MstEdge};

pub use config::{SessionConfig, SessionRefreshPolicy};

/// A live clustering session for append-oriented incremental clustering.
///
/// The current implementation allocates an empty HNSW index and records the
/// configuration and backing source. Calls to [`Self::append`] insert point
/// indices through the public HNSW edge-harvesting path and retain harvested
/// candidate edges for later refresh work. Follow-up roadmap items add refresh,
/// bootstrap, and label-snapshot behaviour.
///
/// ## Concurrency model
///
/// `append` requires `&mut self`, giving the caller exclusive access while the
/// session mutates the live index and pending edge buffer. Read-only accessors
/// remain available through shared references. `_labels` uses `Arc<Vec<usize>>`,
/// so later label snapshots can be shared without blocking the writer.
/// Concurrent read-only access through a shared `RwLock` guard is verified by the
/// `concurrent_readers_observe_consistent_point_count` and
/// `snapshot_version_is_immutable_under_concurrent_readers` tests in `session/tests.rs`.
///
/// ## `pending_edges` memory usage
///
/// `pending_edges` accumulates [`CandidateEdge`] values as points are
/// inserted. Each `CandidateEdge` has a raw field payload of
/// `2 × size_of::<usize>() + size_of::<f32>() + size_of::<u64>()` bytes (two
/// endpoint indices, one `f32` distance, and one `u64` sequence). The struct is
/// then rounded up to its 8-byte alignment, which makes it 32 bytes on 64-bit
/// targets.
/// In the worst case a session that has inserted *N* points
/// will hold at most *N* × *M* edges, where *M* is the HNSW `max_connections`
/// parameter (default 16). For 10 000 points with `M = 16`, the
/// 10 000 × 16 × 32-byte buffer is ~5.12 MB — modest for a transient buffer,
/// but callers must be aware that
/// `pending_edges` grows without bound until a future `refresh()` call
/// (roadmap item 11.1.4) drains it. Long-lived sessions inserting very many
/// points should plan for a periodic refresh cadence or monitor the per-point
/// harvest volume via the `chutoro.session.harvested_edges` counter (enabled
/// with the `metrics` Cargo feature).
///
/// # Examples
/// ```rust,no_run
/// use std::sync::Arc;
///
/// use chutoro_core::{
///     ChutoroBuilder, ChutoroError, ClusteringSession, DataSource,
///     DataSourceError, MetricDescriptor,
/// };
///
/// struct Dummy(Vec<f32>);
///
/// impl DataSource for Dummy {
///     fn len(&self) -> usize { self.0.len() }
///     fn name(&self) -> &str { "dummy" }
///     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
///         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
///         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
///         Ok((a - b).abs())
///     }
///     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("abs") }
/// }
///
/// # fn main() -> Result<(), ChutoroError> {
/// let source = Arc::new(Dummy(vec![0.0, 1.0]));
/// let mut session: ClusteringSession<Dummy> = ChutoroBuilder::new()
///     .build_session(source)?;
///
/// session.append(&[0, 1])?;
///
/// assert_eq!(session.point_count(), 2);
/// assert_eq!(session.snapshot_version(), 0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ClusteringSession<D: DataSource + Send + Sync> {
    config: SessionConfig,
    index: CpuHnsw,
    _core_distances: Vec<f32>,
    _mst_edges: Vec<MstEdge>,
    _historical_edges: Vec<CandidateEdge>,
    pending_edges: Vec<CandidateEdge>,
    _labels: Arc<Vec<usize>>,
    snapshot_version: u64,
    source: Arc<D>,
    _last_refresh_len: usize,
    #[cfg(feature = "metrics")]
    clock: std::sync::Arc<dyn clock::MonotonicClock>,
}

// Verify ClusteringSession is Send + Sync when its DataSource is Send + Sync.
// _DummySrc is Send by default (no non-Send fields); the compiler enforces the
// bound at the call site of assert_send_sync.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}

    struct _DummySrc;

    impl DataSource for _DummySrc {
        fn len(&self) -> usize {
            0
        }

        fn name(&self) -> &str {
            "_dummy"
        }

        fn distance(&self, _i: usize, _j: usize) -> std::result::Result<f32, DataSourceError> {
            Ok(0.0)
        }

        fn metric_descriptor(&self) -> MetricDescriptor {
            MetricDescriptor::new("_dummy")
        }
    }

    assert_send_sync::<ClusteringSession<_DummySrc>>();
};

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    /// Returns the validated configuration used by the session.
    #[must_use]
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Returns the number of points currently inserted into the session.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.index.len()
    }

    /// Returns the most recent published snapshot version.
    #[must_use]
    pub fn snapshot_version(&self) -> u64 {
        self.snapshot_version
    }
}

#[cfg(feature = "metrics")]
mod clock;
mod config;
mod session_impl;

#[cfg(test)]
mod tests;
