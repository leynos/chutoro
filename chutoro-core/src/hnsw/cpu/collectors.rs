//! Edge collectors used by CPU HNSW insertion paths.

use rayon::prelude::*;

use super::{CandidateEdge, CpuHnsw, EdgeHarvest};
use crate::datasource::DataSource;
use crate::hnsw::error::HnswError;

/// Trait for collecting candidate edges during insertion.
///
/// Separates edge harvesting from insertion logic. A collector declares with
/// [`EdgeCollector::WANTS_EDGES`] whether it observes edges at all, so the
/// non-harvesting path can skip `extract_candidate_edges` and its per-insert
/// `Vec` allocation instead of building a vector only to drop it.
pub(super) trait EdgeCollector {
    /// Whether this collector consumes candidate edges extracted from the plan.
    ///
    /// The insertion path is generic over the collector, so this is a
    /// compile-time constant for each monomorphization: `false` lets the
    /// optimiser remove the extraction branch completely.
    const WANTS_EDGES: bool;

    /// Collects edges discovered during a single node insertion.
    fn collect(&mut self, edges: Vec<CandidateEdge>);
}

/// Collector for insertions that do not harvest candidate edges.
///
/// Reports `WANTS_EDGES = false`, so `CpuHnsw::build` never extracts edges
/// for it and allocates no per-insert edge vector. `collect` is retained only
/// to satisfy the shared trait surface; the build path never calls it.
pub(super) struct NoopCollector;

impl EdgeCollector for NoopCollector {
    const WANTS_EDGES: bool = false;

    fn collect(&mut self, _edges: Vec<CandidateEdge>) {}
}

/// Collector that accumulates harvested edges into a `Vec`.
pub(super) struct VecCollector(Vec<CandidateEdge>);

impl VecCollector {
    /// Construct an empty edge collector.
    pub(super) const fn new() -> Self {
        Self(Vec::new())
    }

    /// Return all edges accumulated across insertions.
    pub(super) fn into_inner(self) -> Vec<CandidateEdge> {
        self.0
    }
}

impl EdgeCollector for VecCollector {
    const WANTS_EDGES: bool = true;

    fn collect(&mut self, mut edges: Vec<CandidateEdge>) {
        self.0.append(&mut edges);
    }
}

impl EdgeHarvest {
    /// Collects edges from Rayon-dispatched insertions using `map` → `try_reduce`.
    ///
    /// Each insertion is serialized by the index's insert mutex; Rayon
    /// provides worker dispatch rather than concurrent graph mutation.
    ///
    /// Inserts nodes `1..items` in parallel, accumulating discovered edges into
    /// a single sorted harvest. The returned edges are sorted by insertion
    /// sequence for deterministic ordering.
    pub(super) fn from_parallel_inserts<D: DataSource + Sync>(
        index: &CpuHnsw,
        source: &D,
        items: usize,
    ) -> Result<Self, HnswError> {
        let edges = (1..items)
            .into_par_iter()
            .map(|node| index.insert_with_edges(node, source))
            .try_reduce(Vec::new, |mut acc, node_edges| {
                acc.extend(node_edges);
                Ok(acc)
            })?;

        Ok(Self::from_unsorted(edges))
    }
}

#[cfg(test)]
mod tests {
    //! Pins the compile-time edge requirements declared by each collector.

    use super::{EdgeCollector, NoopCollector, VecCollector};

    /// The build path relies on `NoopCollector` declaring `WANTS_EDGES =
    /// false`: flipping it to `true` reintroduces per-insert edge allocation on
    /// the non-harvesting path. `VecCollector` relies on `true` for the
    /// opposite reason — flipping it to `false` would silently stop harvesting,
    /// because the insertion path would then skip the extraction that feeds
    /// `collect`. Both values are compile-time constants, so the checks live in
    /// `const` blocks.
    ///
    /// This is deliberately a non-behavioural guard: it pins the constants the
    /// insertion path branches on. The insertion behaviour itself is covered by
    /// `non_harvesting_insert_skips_edge_extraction` in `cpu::collector_tests`.
    #[test]
    fn collectors_declare_their_edge_requirements() {
        const {
            assert!(
                !NoopCollector::WANTS_EDGES,
                "NoopCollector must skip edge extraction"
            );
        };
        const {
            assert!(
                VecCollector::WANTS_EDGES,
                "VecCollector must request edge extraction"
            );
        };
    }
}
