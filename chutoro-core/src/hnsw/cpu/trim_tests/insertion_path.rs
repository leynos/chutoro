//! Real-insertion regression for the insertion critical section.
//!
//! The direct scorer test in the parent module scores a synthetic batch and
//! observes where its distance callbacks run. This test drives the production
//! path instead: it seeds an index, then inserts one more node while recording
//! every distance callback that happens after planning has finished.
//!
//! The recording distinguishes the fixed path from the former one. Scoring
//! runs on the thread that holds `insert_mutex`, so every post-planning
//! callback must report the same thread identity, the same Rayon worker, and a
//! `WouldBlock` result from a non-blocking probe of the mutex. A dispatching
//! implementation can satisfy the probe — the mutex is held by the caller — but
//! not the thread identity, which is the property this regression exists to
//! pin.
//!
//! A pass is an observation, not a proof. The source offers one bounded
//! steal-opportunity window, so a dispatching implementation is caught when an
//! idle worker accepts the offer; if none does, the caller can still finish the
//! batch itself. The deterministic guard is the lexical contract in
//! `tests/workflow_contracts/trim_scoring_dispatch_test.py`. The observation
//! state itself lives in `insertion_observation`.

use std::{
    num::NonZeroUsize,
    sync::{Arc, TryLockError},
    thread::ThreadId,
};

use crate::{
    MetricDescriptor,
    datasource::{DataSource, PointPair},
    error::DataSourceError,
    hnsw::{CandidateEdge, CpuHnsw, HnswParams},
};

use super::{
    DenseSource,
    insertion_observation::{CallbackRecord, ObservationState, Probe},
};

/// Small fan-out so a seeded graph fills its adjacency lists quickly and the
/// target insertion therefore produces trim work.
const SMALL_MAX_CONNECTIONS: usize = 4;

/// Construction width large enough to reach several full neighbour lists in
/// one insertion, while still at least the fan-out.
const SMALL_EF_CONSTRUCTION: usize = 8;

/// Nodes inserted before the target insertion.
///
/// Twelve nodes against a fan-out of four leave most base-layer adjacency
/// lists full, so the target insertion stages more than one trim job.
const SEEDED_NODES: usize = 12;

/// Worker threads in the pool the regression installs into.
const INSERTION_THREADS: usize = 4;

/// A collector that marks the post-planning phase without harvesting edges.
struct PhaseMarkingCollector {
    /// Observation the collector marks.
    observation: Arc<ObservationState>,
}

impl PhaseMarkingCollector {
    /// Wraps `observation` in a collector that requests edge extraction.
    fn new(observation: Arc<ObservationState>) -> Self {
        Self { observation }
    }
}

impl super::super::collectors::EdgeCollector for PhaseMarkingCollector {
    /// Edge extraction must run for `collect` to be called at all, and that
    /// call is the phase marker this regression relies on.
    const WANTS_EDGES: bool = true;

    fn collect(&mut self, _edges: Vec<CandidateEdge>) {
        self.observation.mark_post_planning();
    }
}

/// A dense source that records every callback made after planning.
struct InsertionRecordingSource<'index> {
    /// Points and distances used by the insertion.
    points: DenseSource,
    /// Shared observation the callbacks append to.
    observation: Arc<ObservationState>,
    /// Index whose insertion mutex each callback probes.
    index: &'index CpuHnsw,
}

impl<'index> InsertionRecordingSource<'index> {
    /// Wraps `points`, recording into `observation` and probing `index`.
    fn new(
        points: DenseSource,
        index: &'index CpuHnsw,
        observation: Arc<ObservationState>,
    ) -> Self {
        Self {
            points,
            observation,
            index,
        }
    }

    /// Probes the insertion mutex without blocking, then records `callback`.
    ///
    /// The probe is what shows the callback ran inside the critical section;
    /// the recorded thread identity is what shows it ran on the holder.
    fn record(&self, callback: &'static str) {
        let probe = match self.index.insert_mutex.try_lock() {
            Ok(guard) => {
                drop(guard);
                Probe::Acquired
            }
            Err(TryLockError::WouldBlock) => Probe::WouldBlock,
            Err(TryLockError::Poisoned(_)) => Probe::Poisoned,
        };
        self.observation.record(callback, probe);
    }
}

impl DataSource for InsertionRecordingSource<'_> {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn name(&self) -> &'static str {
        "insertion-path-scheduling"
    }

    fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
        self.record("distance");
        self.points.distance(left, right)
    }

    fn distance_batch(&self, pairs: &[PointPair], out: &mut [f32]) -> Result<(), DataSourceError> {
        self.record("distance_batch");
        // Mirror the default implementation: reject a length mismatch before
        // writing, and compute into a temporary so a failed distance leaves
        // `out` untouched. Recording each distance would otherwise be lost.
        if pairs.len() != out.len() {
            return Err(DataSourceError::OutputLengthMismatch {
                out: out.len(),
                expected: pairs.len(),
            });
        }
        let mut computed = vec![0.0_f32; pairs.len()];
        for (pair, slot) in pairs.iter().zip(computed.iter_mut()) {
            *slot = self.distance(pair.left(), pair.right())?;
        }
        out.copy_from_slice(&computed);
        Ok(())
    }

    fn batch_distances(
        &self,
        query: usize,
        candidates: &[usize],
    ) -> Result<Vec<f32>, DataSourceError> {
        self.record("batch_distances");
        if query >= self.points.len() {
            return Err(DataSourceError::OutOfBounds { index: query });
        }
        let pairs: Vec<PointPair> = candidates
            .iter()
            .copied()
            .map(|candidate| PointPair::new(query, candidate))
            .collect();
        let mut out = vec![0.0; pairs.len()];
        self.distance_batch(&pairs, &mut out)?;
        Ok(out)
    }

    fn metric_descriptor(&self) -> MetricDescriptor {
        self.record("metric_descriptor");
        self.points.metric_descriptor()
    }
}

/// Builds the small parameter set that makes an insertion stage trim jobs.
fn small_params() -> HnswParams {
    HnswParams::new(SMALL_MAX_CONNECTIONS, SMALL_EF_CONSTRUCTION)
        .expect("small fan-out parameters must be valid")
        .with_distance_cache_max_entries(
            NonZeroUsize::new(1).expect("one is a non-zero cache capacity"),
        )
}

/// Builds the dedicated pool the observation runs in.
fn dedicated_pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(INSERTION_THREADS)
        .build()
        .expect("a dedicated pool")
}

/// Inserts the seed nodes, resetting the observation so their callbacks are
/// discarded.
fn seed_index(
    index: &CpuHnsw,
    source: &InsertionRecordingSource<'_>,
    collector: &mut PhaseMarkingCollector,
    observation: &ObservationState,
) {
    for node in 0..SEEDED_NODES {
        observation.reset(false);
        index
            .insert_with_collector(node, source, collector)
            .expect("preparatory insertion must succeed");
    }
}

/// Identifies the thread that invoked an insertion and its Rayon worker.
#[derive(Clone, Copy, Debug)]
struct InsertingThread {
    /// Thread that called `insert_with_collector`.
    thread: ThreadId,
    /// Rayon worker index of that thread, when it is a pool worker.
    worker: Option<usize>,
}

impl InsertingThread {
    /// Captures the calling thread, which must be the mutex holder.
    fn capture() -> Self {
        Self {
            thread: std::thread::current().id(),
            worker: rayon::current_thread_index(),
        }
    }
}

/// Asserts the target insertion staged real trim work on a pool worker.
///
/// Without trim work the observation would pass vacuously, and the one-entry
/// cache is what forces that work to reach the distance callbacks.
fn assert_trim_activity(records: &[CallbackRecord], inserting: InsertingThread) {
    let batch_calls = records
        .iter()
        .filter(|record| record.callback == "batch_distances")
        .count();
    assert!(
        batch_calls >= 2,
        concat!(
            "the target insertion must stage at least two trim jobs so the ",
            "regression observes real scoring; it recorded {} batch callbacks",
        ),
        batch_calls
    );
    assert!(
        records.iter().any(|record| record.callback == "distance"),
        "the one-entry cache must force trim scoring to compute a distance"
    );
    assert!(
        inserting.worker.is_some(),
        "the target insertion should have run on a pool worker"
    );
}

/// Asserts the collector and every callback ran on the inserting thread.
///
/// The inserting thread, not the collector's thread, is the reference: an
/// implementation that handed both collection and scoring to another worker
/// would agree with itself while the mutex holder waited.
fn assert_callbacks_on_holder(
    records: &[CallbackRecord],
    observation: &ObservationState,
    inserting: InsertingThread,
) {
    assert_eq!(
        observation.critical_thread(),
        Some(inserting.thread),
        "the edge collector must run on the insert-mutex holder"
    );
    assert_eq!(
        observation.critical_worker(),
        inserting.worker,
        "the edge collector must run on the holder's Rayon worker"
    );
    assert!(
        records
            .iter()
            .all(|record| record.thread == inserting.thread),
        "every post-planning callback must run on the insert-mutex holder"
    );
    assert!(
        records
            .iter()
            .all(|record| record.worker == inserting.worker),
        "every post-planning callback must stay on the holder's Rayon worker"
    );
    assert!(
        records
            .iter()
            .all(|record| record.probe == Probe::WouldBlock),
        "every post-planning callback must observe a held insert mutex"
    );
}

/// Observes trim scoring on a real insertion staying on the mutex holder.
///
/// The collector marks the moment planning ends. Every callback after that
/// mark is inside the insertion critical section, so it must run on the thread
/// that made the insertion, on the same Rayon worker, while `insert_mutex` is
/// held.
///
/// The source forces cache misses by keeping one cache entry, so every trim
/// job reaches the distance callbacks rather than being answered entirely from
/// the cache. A pass is an observation, not a proof; see the module docs.
#[test]
fn insertion_trim_scoring_runs_on_the_insert_mutex_holder() {
    let observation = Arc::new(ObservationState::new());
    let pool = dedicated_pool();

    let inserting = pool.install(|| {
        let inserting = InsertingThread::capture();
        let index = CpuHnsw::with_capacity(small_params(), super::TRIM_POINTS).expect("index");
        let source = InsertionRecordingSource::new(
            DenseSource::synthetic(super::TRIM_POINTS),
            &index,
            Arc::clone(&observation),
        );
        let mut collector = PhaseMarkingCollector::new(Arc::clone(&observation));

        seed_index(&index, &source, &mut collector, &observation);

        // Only the target insertion's callbacks are evidence.
        observation.reset(true);
        index
            .insert_with_collector(SEEDED_NODES, &source, &mut collector)
            .expect("the target insertion must succeed");
        inserting
    });

    let records = observation.records();
    assert_trim_activity(&records, inserting);
    assert_callbacks_on_holder(&records, &observation, inserting);
}
