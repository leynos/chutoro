//! Core-distance recomputation for incremental clustering sessions.
//!
//! The child `helpers` module holds the pure decisions, so it can be tested
//! without an index. The methods here obtain sorted neighbour lists from the
//! HNSW adapter, exclude the query point, and maintain the values the session
//! stores for its points.

mod helpers;

use std::num::NonZeroUsize;

use super::ClusteringSession;
use crate::{DataSource, Neighbour, Result};
use tracing::{instrument, warn};

#[cfg(feature = "metrics")]
use self::helpers::core_distance_error_reason;
pub(super) use self::helpers::recompute_targets;
use self::helpers::{core_distance_from_neighbours, core_updates_for, effective_ef};

impl<D: DataSource + Send + Sync> ClusteringSession<D> {
    /// Mark `index` dirty and grow core-distance storage when needed.
    pub(super) fn mark_core_distance_dirty(&mut self, index: usize) {
        let len = index.saturating_add(1);
        if self.core_distances.len() < len {
            self.core_distances.resize(len, f32::INFINITY);
            self.dirty_core_distances.resize(len, false);
        }
        if let Some(core_distance) = self.core_distances.get_mut(index) {
            *core_distance = f32::INFINITY;
        }
        if let Some(is_dirty) = self.dirty_core_distances.get_mut(index) {
            *is_dirty = true;
        }
    }

    /// Iterate the slots that have been inserted or remain dirty.
    fn inserted_core_distance_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.core_distances
            .iter()
            .zip(&self.dirty_core_distances)
            .enumerate()
            .filter_map(|(index, (distance, is_dirty))| {
                (*is_dirty || distance.is_finite()).then_some(index)
            })
    }

    /// Collect the core-distance slots that require recomputation.
    fn dirty_core_distance_indices(&self) -> Vec<usize> {
        self.dirty_core_distances
            .iter()
            .enumerate()
            .filter_map(|(index, is_dirty)| is_dirty.then_some(index))
            .collect()
    }

    /// Derive the HNSW search width used for core-distance recomputation.
    fn core_distance_ef(&self) -> Option<NonZeroUsize> {
        let point_count = NonZeroUsize::new(self.point_count())?;
        let ef_construction = NonZeroUsize::new(self.config.hnsw_params().ef_construction())?;
        Some(effective_ef(
            self.config.min_cluster_size(),
            ef_construction,
            point_count,
        ))
    }

    /// Search for `point`'s neighbours while excluding the point itself.
    fn search_non_self_neighbours(&self, point: usize, ef: NonZeroUsize) -> Result<Vec<Neighbour>> {
        #[cfg(feature = "metrics")]
        metrics::counter!("chutoro.session.core_distance.queries_total").increment(1);

        let neighbours = self
            .index
            .search(self.source.as_ref(), point, ef)
            .map_err(|error| {
                let chutoro_error = self.map_hnsw_error(error);
                warn!(
                    point,
                    error = ?chutoro_error,
                    "core-distance recompute failed during HNSW search"
                );
                #[cfg(feature = "metrics")]
                metrics::counter!(
                    "chutoro.session.core_distance.errors_total",
                    "reason" => core_distance_error_reason(&chutoro_error)
                )
                .increment(1);
                chutoro_error
            })?;
        Ok(neighbours
            .into_iter()
            .filter(|neighbour| neighbour.id != point)
            .collect())
    }

    /// Derive and store `point`'s core distance from sorted neighbours.
    fn write_core_distance(&mut self, point: usize, neighbours: &[Neighbour]) {
        let core = core_distance_from_neighbours(neighbours, self.config.min_cluster_size());
        self.write_core_distance_value(point, core);
    }

    /// Store a recomputed core distance and clear its dirty flag.
    fn write_core_distance_value(&mut self, point: usize, core: f32) {
        if let Some(core_distance) = self.core_distances.get_mut(point) {
            *core_distance = core;
        }
        if let Some(is_dirty) = self.dirty_core_distances.get_mut(point) {
            *is_dirty = false;
        }
    }

    /// Recomputes dirty and touched core distances.
    ///
    /// Newly appended points are always recomputed. Existing points are
    /// recomputed when they appear in a newly appended point's non-self
    /// neighbour list, which is the cheap incremental approximation used by
    /// roadmap item 11.1.4.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
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
    /// let source = Arc::new(Dummy(vec![0.0, 1.0, 2.0, 10.0]));
    /// let mut session = ChutoroBuilder::new()
    ///     .with_min_cluster_size(1)
    ///     .build_session(source)?;
    ///
    /// session.append(&[0, 1, 2])?;
    /// assert_eq!(session.core_distance(2), None);
    ///
    /// session.recompute_core_distances()?;
    /// assert!(session.core_distance(2).is_some());
    ///
    /// session.append(&[3])?;
    /// session.recompute_core_distances()?;
    /// assert!(session.core_distance(3).is_some());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::ChutoroError::DataSource`] when the backing source
    /// rejects a distance query. Returns
    /// [`crate::ChutoroError::CpuHnswFailure`] when HNSW search reports a
    /// structural failure.
    #[instrument(skip(self), level = "debug")]
    pub fn recompute_core_distances(&mut self) -> Result<()> {
        #[cfg(feature = "metrics")]
        let t0 = self.clock.now();

        let new_indices = self.dirty_core_distance_indices();
        if new_indices.is_empty() {
            #[cfg(feature = "metrics")]
            {
                metrics::histogram!("chutoro.session.core_distance.touched_existing_per_recompute")
                    .record(0.0);
                metrics::histogram!("chutoro.session.core_distance.recompute_seconds")
                    .record(self.clock.now().duration_since(t0).as_secs_f64());
            }
            return Ok(());
        }

        #[cfg(feature = "metrics")]
        metrics::counter!("chutoro.session.core_distance.appends_left_dirty_total").increment(1);

        self.refresh_core_distances(
            &new_indices,
            #[cfg(feature = "metrics")]
            t0,
        )
    }

    /// Recompute core distances for the dirty points and their existing peers.
    fn refresh_core_distances(
        &mut self,
        new_indices: &[usize],
        #[cfg(feature = "metrics")] t0: std::time::Instant,
    ) -> Result<()> {
        let Some(ef) = self.core_distance_ef() else {
            return Ok(());
        };

        let neighbour_lists = self.neighbour_lists_for(new_indices, ef)?;
        let neighbour_slices = neighbour_lists
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let existing_targets = recompute_targets(new_indices, &neighbour_slices);

        #[cfg(feature = "metrics")]
        metrics::histogram!("chutoro.session.core_distance.touched_existing_per_recompute").record(
            u32::try_from(existing_targets.len()).map_or_else(|_| f64::from(u32::MAX), f64::from),
        );

        let mut pending_updates = core_updates_for(
            new_indices,
            &neighbour_lists,
            self.config.min_cluster_size(),
        );
        pending_updates.extend(self.search_core_updates(&existing_targets, ef)?);
        self.apply_core_updates(pending_updates);

        #[cfg(feature = "metrics")]
        {
            let recomputed_existing = u64::try_from(existing_targets.len()).unwrap_or(u64::MAX);
            metrics::counter!("chutoro.session.core_distance.recomputed_existing")
                .increment(recomputed_existing);
            metrics::histogram!("chutoro.session.core_distance.recompute_seconds")
                .record(self.clock.now().duration_since(t0).as_secs_f64());
        }

        Ok(())
    }

    /// Search the neighbours that seed each new point's core distance.
    fn neighbour_lists_for(
        &self,
        points: &[usize],
        ef: NonZeroUsize,
    ) -> Result<Vec<Vec<Neighbour>>> {
        let mut neighbour_lists = Vec::with_capacity(points.len());
        for &point in points {
            neighbour_lists.push(self.search_non_self_neighbours(point, ef)?);
        }
        Ok(neighbour_lists)
    }

    /// Search and derive core distances for existing points touched by inserts.
    fn search_core_updates(&self, points: &[usize], ef: NonZeroUsize) -> Result<Vec<(usize, f32)>> {
        let mut updates = Vec::with_capacity(points.len());
        for &point in points {
            let neighbours = self.search_non_self_neighbours(point, ef)?;
            let core = core_distance_from_neighbours(&neighbours, self.config.min_cluster_size());
            updates.push((point, core));
        }
        Ok(updates)
    }

    /// Store recomputed core distances and clear their dirty flags.
    fn apply_core_updates(&mut self, updates: Vec<(usize, f32)>) {
        for (point, core) in updates {
            self.write_core_distance_value(point, core);
        }
    }

    /// Recomputes every inserted point's core distance.
    ///
    /// This mirrors the batch CPU pipeline's core-distance loop and clears all
    /// dirty cells that correspond to inserted points.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
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
    /// let source = Arc::new(Dummy(vec![0.0, 1.0, 2.0, 10.0]));
    /// let mut session = ChutoroBuilder::new()
    ///     .with_min_cluster_size(2)
    ///     .build_session(source)?;
    ///
    /// session.append(&[0, 1, 2])?;
    /// session.recompute_core_distances()?;
    /// let incremental = session.core_distance(0);
    ///
    /// session.append(&[3])?;
    /// session.recompute_core_distances_full()?;
    /// assert!(session.core_distance(0).is_some());
    /// assert!(session.core_distance(3).is_some());
    /// assert_ne!(incremental, session.core_distance(3));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::ChutoroError::DataSource`] when the backing source
    /// rejects a distance query. Returns
    /// [`crate::ChutoroError::CpuHnswFailure`] when HNSW search reports a
    /// structural failure.
    #[instrument(skip(self), level = "debug")]
    pub fn recompute_core_distances_full(&mut self) -> Result<()> {
        #[cfg(feature = "metrics")]
        let t0 = self.clock.now();

        let Some(ef) = self.core_distance_ef() else {
            return Ok(());
        };
        let indices = self.inserted_core_distance_indices().collect::<Vec<_>>();
        for point in indices {
            let neighbours = self.search_non_self_neighbours(point, ef)?;
            self.write_core_distance(point, &neighbours);
        }

        #[cfg(feature = "metrics")]
        metrics::histogram!("chutoro.session.core_distance.recompute_seconds")
            .record(self.clock.now().duration_since(t0).as_secs_f64());

        Ok(())
    }
}
