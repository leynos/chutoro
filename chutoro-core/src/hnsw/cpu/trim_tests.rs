//! Regression tests for the insertion trim scorer.
//!
//! Trimming runs inside the insertion critical section, so these tests pin the
//! scheduling property that keeps it from stranding the worker that holds the
//! insert mutex.

use std::{collections::BTreeSet, sync::Mutex};

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

/// Rounds of trim scoring the regression performs inside one pool install.
///
/// A dispatching implementation can complete a single round on the calling
/// worker when no other worker steals from it, so one round can pass by
/// coincidence. Repeating the round makes that coincidence implausible without
/// depending on timing.
const TRIM_ROUNDS: usize = 8;

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

/// Wraps a source and records which Rayon worker asked for each distance.
struct ThreadRecordingSource {
    points: DenseSource,
    workers: Mutex<BTreeSet<Option<usize>>>,
}

impl ThreadRecordingSource {
    /// Wraps `points` with a fresh record of the threads that score against it.
    fn new(points: DenseSource) -> Self {
        Self {
            points,
            workers: Mutex::new(BTreeSet::new()),
        }
    }

    /// Returns the distinct worker indices that have called
    /// [`DataSource::distance`].
    fn workers(&self) -> BTreeSet<Option<usize>> {
        self.workers
            .lock()
            .expect("the worker record is never poisoned by this test")
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

    /// Records the calling worker, then forwards to the wrapped source.
    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        let _ = self
            .workers
            .lock()
            .expect("the worker record is never poisoned by this test")
            .insert(rayon::current_thread_index());
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
/// The test scores several rounds of jobs from inside a worker of a four-thread
/// pool, and every distance call across every round must come from that worker.
/// One round is not enough on its own: a dispatching implementation can finish
/// the whole round on the caller when no other worker steals from it, so the
/// round would pass despite the dispatch. Each round uses a fresh index, which
/// keeps the distance cache from turning later rounds into cache hits that never
/// reach the source, and repeats the race enough times that a scheduler that
/// always kept the work on the caller is not a realistic explanation.
#[test]
fn trim_scoring_stays_on_the_thread_holding_the_insert_mutex() {
    let points = DenseSource::synthetic(TRIM_POINTS);
    let source = ThreadRecordingSource::new(points);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(TRIM_THREADS)
        .build()
        .expect("a dedicated pool");

    let (caller, rounds) = pool.install(|| {
        let caller = rayon::current_thread_index();
        let rounds = (0..TRIM_ROUNDS)
            .map(|_| {
                let index =
                    CpuHnsw::with_capacity(HnswParams::new(16, 200).expect("params"), TRIM_POINTS)
                        .expect("index");
                let jobs: Vec<TrimJob> = (0..TRIM_JOBS).map(DenseSource::trim_job).collect();
                index.score_trim_jobs(jobs, &source)
            })
            .collect::<Vec<_>>();
        (caller, rounds)
    });

    for (round, scoring) in rounds.into_iter().enumerate() {
        let scored = scoring.expect("trim scoring must succeed");
        assert_eq!(
            scored.len(),
            TRIM_JOBS,
            "round {round} must score every trim job"
        );
    }
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
