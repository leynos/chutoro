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
};

use crate::{
    MetricDescriptor,
    datasource::{DataSource, PointPair},
    error::DataSourceError,
    hnsw::{CandidateEdge, CpuHnsw, HnswParams},
};

use super::{
    DenseSource,
    insertion_observation::{ObservationState, Probe},
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
        for (pair, slot) in pairs.iter().zip(out.iter_mut()) {
            *slot = self.distance(pair.left(), pair.right())?;
        }
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

/// Observes trim scoring on a real insertion staying on the mutex holder.
///
/// The collector marks the moment planning ends. Every callback after that
/// mark is inside the insertion critical section, so it must run on the thread
/// that set the mark, on the same Rayon worker, while `insert_mutex` is held.
///
/// The source forces cache misses by keeping one cache entry, so every trim
/// job reaches the distance callbacks rather than being answered entirely from
/// the cache. A pass is an observation, not a proof; see the module docs.
#[test]
fn insertion_trim_scoring_runs_on_the_insert_mutex_holder() {
    let observation = Arc::new(ObservationState::new());
    let params = HnswParams::new(SMALL_MAX_CONNECTIONS, SMALL_EF_CONSTRUCTION)
        .expect("small fan-out parameters must be valid")
        .with_distance_cache_max_entries(
            NonZeroUsize::new(1).expect("one is a non-zero cache capacity"),
        );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(INSERTION_THREADS)
        .build()
        .expect("a dedicated pool");

    let (insertion, caller) = pool.install(|| {
        let caller = rayon::current_thread_index();
        let index = CpuHnsw::with_capacity(params, super::TRIM_POINTS).expect("index");
        let source = InsertionRecordingSource::new(
            DenseSource::synthetic(super::TRIM_POINTS),
            &index,
            Arc::clone(&observation),
        );
        let mut collector = PhaseMarkingCollector::new(Arc::clone(&observation));

        for node in 0..SEEDED_NODES {
            observation.reset(false);
            index
                .insert_with_collector(node, &source, &mut collector)
                .expect("preparatory insertion must succeed");
        }

        // Only the target insertion's callbacks are evidence.
        observation.reset(true);
        let insertion = index.insert_with_collector(SEEDED_NODES, &source, &mut collector);
        (insertion, caller)
    });

    insertion.expect("the target insertion must succeed");
    let records = observation.records();
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
        caller.is_some(),
        "the insertion closure should have run on a pool worker"
    );
    assert!(
        records
            .iter()
            .all(|record| Some(record.thread) == observation.critical_thread()),
        "every post-planning callback must run on the insert-mutex holder"
    );
    assert!(
        records
            .iter()
            .all(|record| record.worker == observation.critical_worker()),
        "every post-planning callback must stay on the holder's Rayon worker"
    );
    assert!(
        records
            .iter()
            .all(|record| record.probe == Probe::WouldBlock),
        "every post-planning callback must observe a held insert mutex"
    );
}
