//! Append and construction implementation for [`super::ClusteringSession`].
//!
//! The session struct and accessors remain in `session/mod.rs`; this module
//! holds the mutating behaviour that wires [`super::SessionConfig`],
//! [`crate::CpuHnsw`], the backing [`crate::DataSource`], and optional metrics
//! timing together. That split keeps the public session surface small while
//! preserving the same inherent methods for callers.

use std::sync::Arc;

use super::{ClusteringSession, SessionConfig};
use crate::{ChutoroError, CpuHnsw, DataSource, DataSourceError, HnswError, Result};
use tracing::{debug, instrument, warn};

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    fn append_index_error(&self, index: usize) -> ChutoroError {
        ChutoroError::DataSource {
            data_source: Arc::from(self.source.name()),
            error: DataSourceError::OutOfBounds { index },
        }
    }

    pub(super) fn map_hnsw_error(&self, error: HnswError) -> ChutoroError {
        crate::cpu_pipeline::map_cpu_hnsw_error(self.source.as_ref(), error)
    }

    fn new_with_index_result(
        config: SessionConfig,
        source: Arc<D>,
        index: std::result::Result<CpuHnsw, HnswError>,
    ) -> Result<Self> {
        let index = index.map_err(|error| {
            let code = Arc::from(error.code().as_str());
            let message = Arc::from(error.to_string());
            warn!(
                code = ?code,
                message = %message,
                "CpuHnsw index allocation failed; returning CpuHnswFailure"
            );
            ChutoroError::CpuHnswFailure { code, message }
        })?;
        debug!(
            min_cluster_size = %config.min_cluster_size(),
            "ClusteringSession allocated: empty HNSW index ready"
        );

        #[cfg(feature = "metrics")]
        {
            metrics::describe_counter!(
                "chutoro.session.append.errors_total",
                "Total number of append failures, labelled by reason."
            );
            metrics::describe_histogram!(
                "chutoro.session.append.point_seconds",
                metrics::Unit::Seconds,
                "Per-point HNSW insertion latency in seconds."
            );
            metrics::describe_counter!(
                "chutoro.session.harvested_edges",
                metrics::Unit::Count,
                "Total harvested candidate edges buffered for refresh."
            );
            metrics::describe_counter!(
                "chutoro.session.core_distance.queries_total",
                "Total HNSW searches used for session core-distance recompute."
            );
            metrics::describe_counter!(
                "chutoro.session.core_distance.recomputed_existing",
                "Total existing points recomputed after appearing near new points."
            );
            metrics::describe_counter!(
                "chutoro.session.core_distance.appends_left_dirty_total",
                "Recompute calls that started with one or more dirty core distances."
            );
            metrics::describe_counter!(
                "chutoro.session.core_distance.errors_total",
                "Total number of core-distance recompute failures, labelled by reason."
            );
            metrics::describe_histogram!(
                "chutoro.session.core_distance.touched_existing_per_recompute",
                metrics::Unit::Count,
                "Existing-point fan-out touched by incremental core-distance recompute."
            );
            metrics::describe_histogram!(
                "chutoro.session.core_distance.recompute_seconds",
                metrics::Unit::Seconds,
                "Session core-distance recompute duration in seconds."
            );
        }

        Ok(Self {
            config,
            index,
            core_distances: Vec::with_capacity(source.len()),
            dirty_core_distances: Vec::with_capacity(source.len()),
            _mst_edges: Vec::new(),
            _historical_edges: Vec::new(),
            pending_edges: Vec::new(),
            _labels: Arc::new(Vec::new()),
            snapshot_version: 0,
            source,
            _last_refresh_len: 0,
            #[cfg(feature = "metrics")]
            clock: std::sync::Arc::new(super::clock::StdMonotonicClock),
        })
    }

    fn new_with_capacity(config: SessionConfig, source: Arc<D>, capacity: usize) -> Result<Self> {
        let index = CpuHnsw::with_capacity(config.hnsw_params().clone(), capacity);
        Self::new_with_index_result(config, source, index)
    }

    pub(crate) fn new(config: SessionConfig, source: Arc<D>) -> Result<Self> {
        let capacity = source.len().max(1);
        Self::new_with_capacity(config, source, capacity)
    }

    #[cfg(test)]
    pub(crate) fn new_failing_for_test(config: SessionConfig, source: Arc<D>) -> Result<Self> {
        Self::new_with_index_result(
            config,
            source,
            Err(HnswError::InvalidParameters {
                reason: "test-injected HNSW construction failure".to_owned(),
            }),
        )
    }

    /// Replaces the clock used for per-point latency metrics.
    ///
    /// Only available under `#[cfg(all(feature = "metrics", test))]` so the
    /// production constructor signature is unchanged.
    #[cfg(all(feature = "metrics", test))]
    pub(crate) fn with_clock_for_test(
        mut self,
        clock: std::sync::Arc<dyn super::clock::MonotonicClock>,
    ) -> Self {
        self.clock = clock;
        self
    }

    /// Appends new point indices to the live HNSW index.
    ///
    /// Each index is inserted with [`CpuHnsw::insert_harvesting`], and every
    /// harvested [`crate::CandidateEdge`] is retained in the session's pending
    /// edge buffer for later refresh work. The method is fail-fast: if an
    /// insertion fails, earlier successful insertions and their pending edges
    /// remain in the session.
    ///
    /// # Errors
    ///
    /// Returns [`ChutoroError::DataSource`] when the backing source rejects an
    /// index or distance query. Returns [`ChutoroError::CpuHnswFailure`] when
    /// the HNSW adapter reports a structural failure — including duplicate
    /// indices (detected by [`CpuHnsw::insert_harvesting`] and mapped from
    /// `HnswError::DuplicateIndex`), non-finite distances, lock poisoning, or
    /// graph invariant violations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use chutoro_core::{
    /// #     ChutoroBuilder, ChutoroError, DataSource, DataSourceError,
    /// #     MetricDescriptor,
    /// # };
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    /// #         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    /// #         Ok((a - b).abs())
    /// #     }
    /// #     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("abs") }
    /// # }
    /// # fn main() -> Result<(), ChutoroError> {
    /// let source = Arc::new(Dummy(vec![0.0, 1.0]));
    /// let mut session = ChutoroBuilder::new().build_session(source)?;
    ///
    /// session.append(&[0, 1])?;
    /// assert_eq!(session.point_count(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, indices), fields(count = indices.len()), level = "debug")]
    pub fn append(&mut self, indices: &[usize]) -> Result<()> {
        for &index in indices {
            if index >= self.source.len() {
                warn!(
                    index,
                    source_len = self.source.len(),
                    "append rejected: index out of bounds"
                );
                #[cfg(feature = "metrics")]
                metrics::counter!(
                    "chutoro.session.append.errors_total",
                    "reason" => "out_of_bounds"
                )
                .increment(1);
                return Err(self.append_index_error(index));
            }

            #[cfg(feature = "metrics")]
            let t0 = self.clock.now();

            let edges = self
                .index
                .insert_harvesting(index, self.source.as_ref())
                .map_err(|error| {
                    let chutoro_error = self.map_hnsw_error(error);
                    warn!(
                        index,
                        error = ?chutoro_error,
                        "append rejected: HNSW insertion failed"
                    );
                    #[cfg(feature = "metrics")]
                    metrics::counter!(
                        "chutoro.session.append.errors_total",
                        "reason" => "hnsw_failure"
                    )
                    .increment(1);
                    chutoro_error
                })?;

            #[cfg(feature = "metrics")]
            metrics::histogram!("chutoro.session.append.point_seconds")
                .record(self.clock.now().duration_since(t0).as_secs_f64());

            let harvested = edges.len();
            self.pending_edges.extend(edges);
            self.mark_core_distance_dirty(index);

            #[cfg(feature = "metrics")]
            metrics::counter!("chutoro.session.harvested_edges").increment(harvested as u64);

            debug!(
                index,
                harvested, "append: point inserted and edges buffered"
            );
        }
        Ok(())
    }
}
