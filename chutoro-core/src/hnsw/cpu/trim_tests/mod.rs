//! Regression tests for the insertion trim scorer.
//!
//! Trimming runs inside the insertion critical section, so these tests pin the
//! scheduling property that keeps it from stranding the worker that holds the
//! insert mutex.

use std::{
    collections::BTreeSet,
    sync::{Condvar, Mutex},
    time::Duration,
};

use crate::{
    MetricDescriptor,
    datasource::DataSource,
    error::DataSourceError,
    hnsw::{HnswParams, graph::EdgeContext, insert::TrimJob},
};

use super::CpuHnsw;

/// Dimensions of each point in [`DenseSource`].
const TRIM_DIMENSIONS: usize = 16;

/// Points available to the trim-scoring regression.
const TRIM_POINTS: usize = 512;

/// Trim jobs scored in one call.
///
/// The regression scores enough jobs that a nested parallel iterator would
/// spread them over the pool's idle workers; a handful of jobs could run
/// entirely on the calling thread and hide a regression.
const TRIM_JOBS: usize = 64;

/// Candidates scored for each trim job.
const TRIM_CANDIDATES: usize = 32;

/// Worker threads in the pool the trim-scoring regression installs into.
const TRIM_THREADS: usize = 4;

/// How long the caller's first distance call waits for another worker.
///
/// The wait is what makes the scheduling check deterministic. A dispatching
/// implementation has this long to let an idle worker steal a job and record
/// itself; a sequential implementation waits out the whole interval once and
/// then scores normally.
const OTHER_WORKER_WAIT: Duration = Duration::from_secs(2);

/// Guards the arithmetic in [`DenseSource::trim_job`]: every job scores
/// candidates drawn from `node + offset` without wrapping.
const _: () = assert!(TRIM_JOBS + TRIM_CANDIDATES <= TRIM_POINTS);

/// A dense, fixed-width source for the trim-scoring regression.
///
/// Points are spread pseudo-randomly through a sixteen-dimensional cube so the
/// distance work is representative of a real embedding, which is what leaves an
/// idle worker ready to steal a nested job. The regression does not depend on
/// particular distances, only on where the work runs.
struct DenseSource {
    coordinates: Vec<f32>,
    points: usize,
}

impl DenseSource {
    /// Generates `points` deterministic pseudo-random points in `[1, 2)`.
    fn synthetic(points: usize) -> Self {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let coordinates = (0..points * TRIM_DIMENSIONS)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let bits = u32::try_from(state >> 40).expect("the top 24 bits fit a u32");
                f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000)
            })
            .collect();
        Self {
            coordinates,
            points,
        }
    }

    /// Returns the coordinates of one point, or an error for a bad index.
    fn point(&self, index: usize) -> Result<&[f32], DataSourceError> {
        let start = index * TRIM_DIMENSIONS;
        self.coordinates
            .get(start..start + TRIM_DIMENSIONS)
            .ok_or(DataSourceError::OutOfBounds { index })
    }

    /// Builds one trim job whose candidates overlap its neighbours' windows.
    fn trim_job(job: usize) -> TrimJob {
        TrimJob {
            node: job,
            ctx: EdgeContext {
                level: 0,
                max_connections: 8,
            },
            candidates: (0..TRIM_CANDIDATES).map(|offset| job + offset).collect(),
            sequences: (0..TRIM_CANDIDATES)
                .map(|offset| u64::try_from(offset).expect("the candidate count fits a u64"))
                .collect(),
        }
    }
}

impl DataSource for DenseSource {
    /// Returns the number of points in the synthetic source.
    fn len(&self) -> usize {
        self.points
    }

    /// Identifies the synthetic source in diagnostics.
    fn name(&self) -> &'static str {
        "trim-scoring-regression"
    }

    /// Returns the squared Euclidean distance between two points.
    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let left_point = self.point(left)?;
        let right_point = self.point(right)?;
        Ok(left_point
            .iter()
            .zip(right_point)
            .fold(0.0_f32, |total, (a, b)| {
                let delta = a.mul_add(1.0, std::ops::Neg::neg(*b));
                delta.mul_add(delta, total)
            }))
    }

    /// Reports the squared Euclidean metric the synthetic source simulates.
    fn metric_descriptor(&self) -> MetricDescriptor {
        MetricDescriptor::new("euclidean-squared")
    }
}

/// Shared state for [`ThreadRecordingSource`].
struct Recording {
    /// Distinct Rayon workers that have asked for a distance.
    workers: BTreeSet<Option<usize>>,
    /// The pool worker the test will score from, once it is known.
    caller: Option<usize>,
    /// Whether a worker other than the caller has recorded a call.
    other_worker_seen: bool,
    /// Whether the caller has already served its one bounded wait.
    waited: bool,
}

/// Wraps a source and records which Rayon worker asked for each distance.
struct ThreadRecordingSource {
    points: DenseSource,
    recording: Mutex<Recording>,
    other_worker: Condvar,
}

impl ThreadRecordingSource {
    /// Wraps `points` with a fresh record of the threads that score against it.
    fn new(points: DenseSource) -> Self {
        Self {
            points,
            recording: Mutex::new(Recording {
                workers: BTreeSet::new(),
                caller: None,
                other_worker_seen: false,
                waited: false,
            }),
            other_worker: Condvar::new(),
        }
    }

    /// Tells the source which worker will score, enabling the bounded wait.
    fn set_caller(&self, caller: Option<usize>) {
        self.recording
            .lock()
            .expect("the worker record is never poisoned by this test")
            .caller = caller;
    }

    /// Returns the distinct worker indices that have called
    /// [`DataSource::distance`].
    fn workers(&self) -> BTreeSet<Option<usize>> {
        self.recording
            .lock()
            .expect("the worker record is never poisoned by this test")
            .workers
            .clone()
    }
}

impl DataSource for ThreadRecordingSource {
    /// Returns the number of points in the wrapped source.
    fn len(&self) -> usize {
        self.points.len()
    }

    /// Identifies the wrapped source in diagnostics.
    fn name(&self) -> &'static str {
        "trim-scoring-regression"
    }

    /// Records the calling worker, giving the pool one bounded chance to steal
    /// work before a dispatching implementation can finish on the caller.
    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let worker = rayon::current_thread_index();
        let mut recording = self
            .recording
            .lock()
            .expect("the worker record is never poisoned by this test");
        recording.workers.insert(worker);
        if recording
            .caller
            .is_some_and(|caller| Some(caller) != worker)
        {
            recording.other_worker_seen = true;
            self.other_worker.notify_all();
        } else if recording.caller.is_some() && !recording.waited {
            // The first call on the caller blocks until another worker takes a
            // job or the interval expires. Waiting on every call would pay the
            // interval once per distance; waiting on none of them would leave
            // detection to work stealing alone.
            recording.waited = true;
            let (guard, _timeout) = self
                .other_worker
                .wait_timeout_while(recording, OTHER_WORKER_WAIT, |state| {
                    !state.other_worker_seen
                })
                .expect("the worker record is never poisoned by this test");
            drop(guard);
            return self.points.distance(left, right);
        }
        drop(recording);
        self.points.distance(left, right)
    }

    /// Reports the metric advertised by the wrapped source.
    fn metric_descriptor(&self) -> MetricDescriptor {
        self.points.metric_descriptor()
    }
}

/// Verifies trim scoring never hands work to another Rayon worker.
///
/// Trimming runs while the caller holds `insert_mutex`, so every other worker in
/// a parallel build may already be blocked acquiring that same mutex. Scoring
/// used to be an `into_par_iter()` over the staged jobs, which let Rayon park
/// the mutex holder on a latch that only a blocked worker could satisfy; the
/// build then hung instead of returning, and a benchmark child that hit it had
/// to be killed after 25 minutes.
///
/// The check is deterministic rather than probabilistic: the caller's first
/// distance call waits up to [`OTHER_WORKER_WAIT`] for a call from another
/// worker. A dispatching implementation hands the idle pool an opportunity to
/// steal and fails; a sequential one waits the interval out once and completes
/// on the caller. Repeating rounds is unnecessary, because the wait supplies
/// the opportunity that timing used to supply by chance.
#[test]
fn trim_scoring_stays_on_the_thread_holding_the_insert_mutex() {
    let source = ThreadRecordingSource::new(DenseSource::synthetic(TRIM_POINTS));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(TRIM_THREADS)
        .build()
        .expect("a dedicated pool");

    let (caller, scored) = pool.install(|| {
        let caller = rayon::current_thread_index();
        source.set_caller(caller);
        let index = CpuHnsw::with_capacity(HnswParams::new(16, 200).expect("params"), TRIM_POINTS)
            .expect("index");
        let jobs: Vec<TrimJob> = (0..TRIM_JOBS).map(DenseSource::trim_job).collect();
        (caller, index.score_trim_jobs(jobs, &source))
    });

    let scored_jobs = scored.expect("trim scoring must succeed");
    assert_eq!(
        scored_jobs.len(),
        TRIM_JOBS,
        "every trim job must be scored"
    );
    assert!(
        caller.is_some(),
        "the scoring closure should have run on a pool worker"
    );
    assert_eq!(
        source.workers(),
        BTreeSet::from([caller]),
        concat!(
            "trim scoring dispatched distance work to other Rayon workers, ",
            "where it can strand the insert mutex holder on a latch those ",
            "workers cannot reach"
        )
    );
}
/// Metric assertions for the trim-scoring boundary.
///
/// They live in their own file so the scheduling regression and the recorder
/// setup stay within the module size the lint suite allows.
#[cfg(feature = "metrics")]
mod metric_recording;
