//! Observation state shared by the insertion-path scheduling regression.
//!
//! `insertion_path` drives a real insertion and records where its post-planning
//! distance callbacks run. This module holds the state those callbacks write:
//! the phase they are filtered by, the thread and Rayon worker that served
//! them, the non-blocking `insert_mutex` probe taken beside each call, and the
//! one bounded wait offered to the pool.
//!
//! Splitting it out keeps `insertion_path` inside the 400-line module limit
//! without moving the assertion it supports. Only the sibling module consumes
//! these items, so they are `pub(super)` rather than crate-visible.

use std::{
    sync::{Condvar, Mutex},
    thread::ThreadId,
};

use super::OTHER_WORKER_WAIT;

/// One non-blocking probe of `insert_mutex`, taken while a callback runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Probe {
    /// The mutex was free. A post-planning callback must never observe this.
    Acquired,
    /// The mutex is held, as it is for every callback in the critical section.
    WouldBlock,
    /// The mutex is poisoned. Reported rather than hidden so the assertion
    /// below fails with the reason the probe actually saw.
    Poisoned,
}

/// One distance callback recorded after planning completed.
#[derive(Clone, Copy, Debug)]
pub(super) struct CallbackRecord {
    /// Which `DataSource` method was called.
    pub(super) callback: &'static str,
    /// Thread that served the callback.
    pub(super) thread: ThreadId,
    /// Rayon worker index of that thread, when it is a pool worker.
    pub(super) worker: Option<usize>,
    /// Result of the non-blocking `insert_mutex` probe taken by the callback.
    pub(super) probe: Probe,
}

/// Which insertion's callbacks the observation keeps, and from which phase.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    /// A preparatory insertion: callbacks are ignored.
    #[default]
    Idle,
    /// The target insertion, before the collector marks planning complete.
    Preparing,
    /// The target insertion, after planning: callbacks are evidence.
    Scoring,
}

/// Progress of the bounded steal-opportunity offered to the pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum StealWindow {
    /// The caller has not served its one wait yet.
    #[default]
    Open,
    /// The caller has already waited; later callbacks must not wait again.
    Waited,
    /// Some callback has run on a thread other than the critical thread.
    OtherThreadSeen,
}

/// State shared by the recording source and the phase-marking collector.
#[derive(Default)]
struct Observation {
    /// Which insertion and phase the observation is in.
    phase: Phase,
    /// Thread that ran the collector, which is the `insert_mutex` holder.
    critical_thread: Option<ThreadId>,
    /// Rayon worker index of that thread.
    critical_worker: Option<usize>,
    /// Callbacks recorded after the phase marker was set.
    records: Vec<CallbackRecord>,
    /// Progress of the one bounded wait offered to the pool.
    steal: StealWindow,
}

/// Shared observation plus the condition variable used for the bounded wait.
pub(super) struct ObservationState {
    /// Mutable observation guarded by the mutex.
    state: Mutex<Observation>,
    /// Signalled when a callback runs on a different thread.
    other_thread: Condvar,
}

impl ObservationState {
    /// Creates an observation that records nothing until it is reset.
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(Observation::default()),
            other_thread: Condvar::new(),
        }
    }

    /// Clears the previous insertion's evidence and sets whether to record.
    ///
    /// Called between insertions so a preparatory insertion's callbacks cannot
    /// satisfy the target insertion's assertions.
    pub(super) fn reset(&self, recording: bool) {
        let mut state = self.lock();
        *state = Observation {
            phase: if recording {
                Phase::Preparing
            } else {
                Phase::Idle
            },
            ..Observation::default()
        };
    }

    /// Marks the point at which planning has finished.
    ///
    /// The collector runs while `insert_mutex` is held, so the current thread
    /// is the critical-section thread for every callback that follows.
    pub(super) fn mark_post_planning(&self) {
        let mut state = self.lock();
        state.phase = Phase::Scoring;
        state.critical_thread = Some(std::thread::current().id());
        state.critical_worker = rayon::current_thread_index();
    }

    /// Records one post-planning callback and offers the pool one chance to
    /// steal work before the caller's first callback continues.
    pub(super) fn record(&self, callback: &'static str, probe: Probe) {
        let thread = std::thread::current().id();
        let worker = rayon::current_thread_index();
        let mut state = self.lock();
        if state.phase != Phase::Scoring {
            return;
        }
        state.records.push(CallbackRecord {
            callback,
            thread,
            worker,
            probe,
        });
        if state
            .critical_thread
            .is_some_and(|critical| critical != thread)
        {
            state.steal = StealWindow::OtherThreadSeen;
            self.other_thread.notify_all();
            return;
        }
        if state.steal == StealWindow::Open {
            // The wait is the observation: a dispatching scorer parks the
            // caller here while an idle worker steals a nested job. Waiting on
            // every callback would pay the interval once per distance.
            state.steal = StealWindow::Waited;
            let (_guard, _timeout) = self
                .other_thread
                .wait_timeout_while(state, OTHER_WORKER_WAIT, |observation| {
                    !matches!(observation.steal, StealWindow::OtherThreadSeen)
                })
                .expect("the observation mutex is never poisoned by this test");
        }
    }

    /// Returns the callbacks recorded for the target insertion.
    pub(super) fn records(&self) -> Vec<CallbackRecord> {
        self.lock().records.clone()
    }

    /// Returns the thread that ran the collector, if one has run.
    pub(super) fn critical_thread(&self) -> Option<ThreadId> {
        self.lock().critical_thread
    }

    /// Returns the Rayon worker that ran the collector, if one has run.
    pub(super) fn critical_worker(&self) -> Option<usize> {
        self.lock().critical_worker
    }

    /// Locks the observation, surfacing poison as a test failure.
    fn lock(&self) -> std::sync::MutexGuard<'_, Observation> {
        self.state
            .lock()
            .expect("the observation mutex is never poisoned by this test")
    }
}
