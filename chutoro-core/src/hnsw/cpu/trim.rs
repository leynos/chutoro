//! Trimming logic applied during insertion.

use std::collections::BinaryHeap;

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
    /// Scores trim jobs one at a time, validating candidate, sequence, and
    /// distance lengths before emitting ranked neighbour lists capped at each
    /// edge context's `max_connections`.
    ///
    /// The caller supplies trimmed candidates gathered while the graph lock is
    /// held. This method then validates the batched distances without the lock
    /// and deterministically orders neighbours by distance and insertion
    /// sequence so ties remain stable while a bounded binary heap retains only
    /// the best `max_connections` entries.
    ///
    /// The scoring behaviour is exercised by
    /// `score_trim_jobs_keeps_the_nearest_candidates` in the CPU unit tests,
    /// because this item is crate-private and so cannot appear in a doctest
    /// compiled as an external crate.
    pub(crate) fn score_trim_jobs<D: DataSource + Sync>(
        &self,
        trim_jobs: Vec<TrimJob>,
        source: &D,
    ) -> Result<Vec<TrimResult>, HnswError> {
        if trim_jobs.is_empty() {
            return Ok(Vec::new());
        }

        // Scoring runs inside the insertion critical section, so every other
        // Rayon worker may already be blocked waiting for `insert_mutex`.
        // Dispatching these jobs to the pool lets Rayon park this thread on a
        // latch that only a blocked worker could satisfy, which hung
        // `CpuHnsw::build_with_edges` outright. Trim jobs are few — at most one
        // per staged neighbour update — so scoring them here in order costs far
        // less than the stall it prevents.
        trim_jobs
            .into_iter()
            .map(|job| self.run_trim_job(job, source))
            .collect()
    }

    /// Scores one staged trim batch at the insertion boundary and records its
    /// cost.
    ///
    /// Insertion calls this rather than [`Self::score_trim_jobs`], which stays
    /// a pure scoring unit. The observation belongs here because the caller is
    /// the point that knows the insertion critical section is held, and that
    /// held section is what the instruments exist to show.
    ///
    /// The batch is recorded whether scoring succeeds or fails, because a
    /// failed batch spends the same time inside the critical section. An empty
    /// batch records nothing: it costs nothing to score, and counting it would
    /// describe work the critical section never did.
    ///
    /// The counter counts jobs submitted, not jobs completed. Scoring stops at
    /// the first invalid job, so a count taken afterwards would under-report
    /// the work the boundary was asked to do.
    pub(crate) fn score_trim_jobs_with_metrics<D: DataSource + Sync>(
        &self,
        trim_jobs: Vec<TrimJob>,
        source: &D,
    ) -> Result<Vec<TrimResult>, HnswError> {
        #[cfg(feature = "metrics")]
        let submitted_jobs = trim_jobs.len();
        #[cfg(feature = "metrics")]
        let started = std::time::Instant::now();
        let scored = self.score_trim_jobs(trim_jobs, source);
        #[cfg(feature = "metrics")]
        record_trim_scoring(submitted_jobs, started.elapsed());
        scored
    }

    /// Score and retain the best candidates for one staged trim job.
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
        for ((id, sequence), distance) in candidates.into_iter().zip(sequences).zip(distances) {
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

/// Records one trim batch's submitted job count and wall time.
///
/// Called from the insertion boundary for every non-empty batch, whether
/// scoring succeeded or failed. The job count is the batch size handed to
/// [`CpuHnsw::score_trim_jobs`], not the number that completed, so the counter
/// name says `submitted` to make that distinction explicit.
#[cfg(feature = "metrics")]
fn record_trim_scoring(submitted_jobs: usize, elapsed: std::time::Duration) {
    if submitted_jobs == 0 {
        return;
    }
    metrics::counter!("chutoro.hnsw.trim.jobs_submitted_total").increment(submitted_jobs as u64);
    metrics::histogram!("chutoro.hnsw.trim.duration_seconds").record(elapsed.as_secs_f64());
}
