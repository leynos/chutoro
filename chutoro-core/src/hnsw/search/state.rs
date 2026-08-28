//! Queue state and deterministic ordering for one HNSW layer search.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
};

use crate::hnsw::types::Neighbour;

/// Tracks discovery, candidates, and retained results during layer search.
#[derive(Debug)]
pub(super) struct SearchState {
    /// Nodes already expanded from the candidate queue.
    visited: HashSet<usize>,
    /// Candidates ordered nearest-first for expansion.
    candidates: BinaryHeap<CandidateNeighbour>,
    /// Retained neighbours ordered furthest-first for capacity trimming.
    best: BinaryHeap<BestNeighbour>,
    /// Nodes already admitted to the search frontier.
    discovered: HashSet<usize>,
}

impl SearchState {
    /// Initialise queues using the compatibility fallback capacity.
    pub(super) fn new(entry: SearchNeighbour) -> Self {
        // Fallback when `ef` is not available at the call-site.
        Self::with_capacity(entry, 64)
    }

    /// Initialise queues sized for the requested search width.
    pub(super) fn with_capacity(entry: SearchNeighbour, ef: usize) -> Self {
        let queue_capacity = ef.max(1);
        let set_capacity = queue_capacity.saturating_mul(4);
        let visited = HashSet::with_capacity(set_capacity);
        let mut candidates = BinaryHeap::with_capacity(queue_capacity);
        candidates.push(CandidateNeighbour(entry));
        let mut best = BinaryHeap::with_capacity(queue_capacity);
        best.push(BestNeighbour(entry));
        let mut discovered = HashSet::with_capacity(set_capacity);
        discovered.insert(entry.id);
        Self {
            visited,
            candidates,
            best,
            discovered,
        }
    }

    /// Remove the nearest candidate awaiting expansion.
    pub(super) fn pop_candidate(&mut self) -> Option<SearchNeighbour> {
        self.candidates
            .pop()
            .map(|CandidateNeighbour(neighbour)| neighbour)
    }

    /// Report whether a full result set cannot improve beyond the candidate.
    pub(super) fn should_terminate(&self, ef: usize, candidate_distance: f32) -> bool {
        if self.best.len() < ef {
            return false;
        }
        self.best
            .peek()
            .is_some_and(|BestNeighbour(furthest)| candidate_distance >= furthest.distance)
    }

    /// Mark a candidate as processed, returning whether it was fresh.
    pub(super) fn mark_processed(&mut self, candidate: usize) -> bool {
        self.visited.insert(candidate)
    }

    /// Record a discovered node, returning whether it was newly discovered.
    pub(super) fn discover(&mut self, candidate: usize) -> bool {
        self.discovered.insert(candidate)
    }

    /// Admit a competitive candidate and trim retained results to `ef`.
    pub(super) fn try_enqueue(&mut self, candidate: SearchNeighbour, ef: usize) {
        let id = candidate.id;
        if self.visited.contains(&id) {
            return;
        }
        if self.best.len() >= ef
            && self
                .best
                .peek()
                .is_some_and(|BestNeighbour(furthest)| candidate.distance >= furthest.distance)
        {
            return;
        }
        self.candidates.push(CandidateNeighbour(candidate));
        self.best.push(BestNeighbour(candidate));
        self.enforce_capacity(ef);
    }

    /// Remove furthest retained neighbours beyond the configured width.
    fn enforce_capacity(&mut self, ef: usize) {
        while self.best.len() > ef {
            self.best.pop();
        }
    }

    /// Convert retained queue entries into ascending public neighbours.
    pub(super) fn finalise(self) -> Vec<Neighbour> {
        let mut neighbours: Vec<_> = self.best.into_vec();
        neighbours.sort_unstable();
        neighbours
            .into_iter()
            .map(|BestNeighbour(neighbour)| neighbour.into_public())
            .collect()
    }
}

/// Search-local neighbour ordered with a deterministic insertion sequence.
#[derive(Clone, Copy, Debug)]
pub(super) struct SearchNeighbour {
    /// Node identifier.
    pub(super) id: usize,
    /// Validated distance from the query node.
    pub(super) distance: f32,
    /// Insertion order used to break otherwise equal ties.
    sequence: u64,
}

impl SearchNeighbour {
    /// Build a neighbour snapshot for the search queues.
    pub(super) const fn new(id: usize, distance: f32, sequence: u64) -> Self {
        Self {
            id,
            distance,
            sequence,
        }
    }

    /// Convert this search-local neighbour into the public result type.
    const fn into_public(self) -> Neighbour {
        Neighbour {
            id: self.id,
            distance: self.distance,
        }
    }
}

/// Compare neighbours by distance, node identifier, then insertion sequence.
fn compare_neighbours(left: &SearchNeighbour, right: &SearchNeighbour) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.id.cmp(&right.id))
        .then_with(|| left.sequence.cmp(&right.sequence))
}

/// Implement total ordering for a tuple wrapper around a search neighbour.
macro_rules! impl_neighbour_wrapper {
    ($name:ident, $cmp:expr) => {
        impl Eq for $name {}

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                $cmp(&self.0, &other.0) == Ordering::Equal
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                $cmp(&self.0, &other.0)
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
    };
}

/// Candidate-queue wrapper that reverses nearest-neighbour ordering.
#[derive(Clone, Copy, Debug)]
struct CandidateNeighbour(SearchNeighbour);

impl_neighbour_wrapper!(
    CandidateNeighbour,
    |left: &SearchNeighbour, right: &SearchNeighbour| { compare_neighbours(right, left) }
);

/// Result-queue wrapper that exposes the furthest retained neighbour first.
#[derive(Clone, Copy, Debug)]
struct BestNeighbour(SearchNeighbour);

impl_neighbour_wrapper!(BestNeighbour, compare_neighbours);
