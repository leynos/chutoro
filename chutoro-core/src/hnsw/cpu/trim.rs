//! Trimming logic applied during insertion.

use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::{
    DataSource,
    hnsw::{
        error::HnswError,
        helpers::batch_distances_for_trim,
        insert::{TrimJob, TrimResult},
        params::connection_limit_for_level,
        types::RankedNeighbour,
    },
};

use super::CpuHnsw;

impl CpuHnsw {
    /// Scores trim jobs in parallel, validating candidate, sequence, and
    /// distance lengths before emitting ranked neighbour lists capped at each
    /// edge context's `max_connections`.
    ///
    /// The caller supplies trimmed candidates gathered while the graph lock is
    /// held. This method then validates the batched distances without the lock
    /// and deterministically orders neighbours by distance and insertion
    /// sequence so ties remain stable while a bounded binary heap retains only
    /// the best `max_connections` entries.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use chutoro_core::{
    ///     CpuHnsw,
    ///     DataSource,
    ///     DataSourceError,
    ///     HnswParams,
    ///     hnsw::graph::EdgeContext,
    ///     hnsw::insert::executor::TrimJob,
    /// };
    ///
    /// # struct Dummy(Vec<f32>);
    /// # impl DataSource for Dummy {
    /// #     fn len(&self) -> usize { self.0.len() }
    /// #     fn name(&self) -> &str { "dummy" }
    /// #     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
    /// #         let a = self.0.get(i).ok_or(DataSourceError::OutOfBounds { index: i })?;
    /// #         let b = self.0.get(j).ok_or(DataSourceError::OutOfBounds { index: j })?;
    /// #         Ok((a - b).abs())
    /// #     }
    /// # }
    /// let params = HnswParams::new(1, 2).unwrap();
    /// let hnsw = CpuHnsw::with_capacity(params, 2).unwrap();
    /// let trim_jobs = vec![TrimJob {
    ///     node: 0,
    ///     ctx: EdgeContext { level: 0, max_connections: 1 },
    ///     candidates: vec![0],
    ///     sequences: vec![0],
    /// }];
    /// let results = hnsw.score_trim_jobs(trim_jobs, &Dummy(vec![0.0])).unwrap();
    /// assert_eq!(results[0].neighbours, vec![0]);
    /// ```
    pub(crate) fn score_trim_jobs<D: DataSource + Sync>(
        &self,
        trim_jobs: Vec<TrimJob>,
        source: &D,
    ) -> Result<Vec<TrimResult>, HnswError> {
        if trim_jobs.is_empty() {
            return Ok(Vec::new());
        }

        trim_jobs
            .into_par_iter()
            .map(|job| self.run_trim_job(job, source))
            .collect::<Result<Vec<_>, HnswError>>()
    }

    fn run_trim_job<D: DataSource + Sync>(
        &self,
        job: TrimJob,
        source: &D,
    ) -> Result<TrimResult, HnswError> {
        let TrimJob {
            node,
            ctx,
            candidates,
            sequences,
        } = job;

        let connection_limit = connection_limit_for_level(ctx.level, ctx.max_connections);

        if candidates.len() != sequences.len() {
            return Err(HnswError::InvalidParameters {
                reason: format!(
                    "trim job candidates ({}) must match sequence count ({})",
                    candidates.len(),
                    sequences.len()
                ),
            });
        }

        let distances = batch_distances_for_trim(&self.distance_cache, node, &candidates, source)?;
        if distances.len() != candidates.len() {
            return Err(HnswError::InvalidParameters {
                reason: format!(
                    "trim job distance count ({}) mismatches candidates ({})",
                    distances.len(),
                    candidates.len()
                ),
            });
        }

        let mut heap = BinaryHeap::with_capacity(connection_limit);
        for (index, id) in candidates.into_iter().enumerate() {
            let sequence = sequences[index];
            let distance = distances[index];
            heap.push(RankedNeighbour::new(id, distance, sequence));
            if heap.len() > connection_limit {
                heap.pop();
            }
        }

        let neighbours = heap
            .into_sorted_vec()
            .into_iter()
            .map(|neighbour| neighbour.into_neighbour().id)
            .collect();

        Ok(TrimResult {
            node,
            ctx,
            neighbours,
        })
    }
}
