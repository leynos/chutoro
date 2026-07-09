//! Edge collectors used by CPU HNSW insertion paths.

use rayon::prelude::*;

use super::{CandidateEdge, CpuHnsw, EdgeHarvest};
use crate::datasource::DataSource;
use crate::hnsw::error::HnswError;

/// Trait for collecting candidate edges during insertion.
///
/// Enables separation of edge harvesting from insertion logic, allowing
/// non-harvesting paths to avoid allocation overhead.
pub(super) trait EdgeCollector {
    /// Collects edges discovered during a single node insertion.
    fn collect(&mut self, edges: Vec<CandidateEdge>);
}

/// No-op collector that discards edges without allocation.
pub(super) struct NoopCollector;

impl EdgeCollector for NoopCollector {
    fn collect(&mut self, _edges: Vec<CandidateEdge>) {}
}

/// Collector that accumulates edges into a `Vec`.
pub(super) struct VecCollector(Vec<CandidateEdge>);

impl VecCollector {
    pub(super) fn new() -> Self {
        Self(Vec::new())
    }

    pub(super) fn into_inner(self) -> Vec<CandidateEdge> {
        self.0
    }
}

impl EdgeCollector for VecCollector {
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
