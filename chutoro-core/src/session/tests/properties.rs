//! Property tests for session invariants.

use std::sync::Arc;

use proptest::prelude::*;

use super::common::SessionTestSource;
use crate::{CandidateEdge, ChutoroBuilder, ChutoroError, CpuHnsw, DataSource, HnswParams};

proptest! {
    /// Any non-zero `min_cluster_size` in `[1, 1000]` must yield a session whose
    /// `config().min_cluster_size()` equals the requested value.
    #[test]
    fn build_session_preserves_min_cluster_size(size in 1usize..=1000) {
        let source = Arc::new(SessionTestSource::with_len(0));
        let session = ChutoroBuilder::new()
            .with_min_cluster_size(size)
            .build_session(source)
            .expect("must build for any non-zero min_cluster_size");
        prop_assert_eq!(session.config().min_cluster_size().get(), size);
    }

    /// A freshly-built session must always start with `point_count() == 0` and
    /// `snapshot_version() == 0`, regardless of source length.
    #[test]
    fn build_session_always_starts_empty(len in 0usize..=256) {
        let source = Arc::new(SessionTestSource::with_len(len));
        let session = ChutoroBuilder::new()
            .build_session(source)
            .expect("must build for any source length");
        prop_assert_eq!(session.point_count(), 0);
        prop_assert_eq!(session.snapshot_version(), 0);
    }

    /// Zero `min_cluster_size` must always return `InvalidMinClusterSize`,
    /// regardless of any other builder state.
    #[test]
    fn build_session_always_rejects_zero_min_cluster_size(len in 0usize..=256) {
        let source = Arc::new(SessionTestSource::with_len(len));
        let err = ChutoroBuilder::new()
            .with_min_cluster_size(0)
            .build_session(source)
            .expect_err("zero must always fail");
        prop_assert!(
            matches!(err, ChutoroError::InvalidMinClusterSize { got: 0 }),
            "expected InvalidMinClusterSize {{ got: 0 }}, got {err:?}"
        );
    }

    /// Appending a generated unique sequence of in-bounds indices must make
    /// `point_count()` match the number of appended points.
    #[test]
    fn append_unique_indices_updates_point_count(
        indices in proptest::collection::vec(0usize..32, 0..=32)
            .prop_map(|mut indices| {
                indices.sort_unstable();
                indices.dedup();
                indices
            })
    ) {
        let source = Arc::new(SessionTestSource::with_len(32));
        let mut session = ChutoroBuilder::new()
            .with_hnsw_params(HnswParams::default().with_rng_seed(73))
            .build_session(source)
            .expect("session must build");

        session
            .append(&indices)
            .expect("unique in-bounds append sequence must succeed");

        prop_assert_eq!(session.point_count(), indices.len());
        prop_assert_eq!(session.snapshot_version(), 0);
    }

    /// For any unique, in-bounds index sequence of length ≥ 1, `append`
    /// must buffer exactly the candidate edges that direct
    /// `CpuHnsw::insert_harvesting` produces for the same sequence.
    #[test]
    fn append_pending_edges_match_direct_harvested_edges(
        indices in proptest::collection::vec(0usize..16, 1..=16)
            .prop_map(|v| {
                let mut seen = std::collections::HashSet::new();
                v.into_iter()
                    .filter(|index| seen.insert(*index))
                    .collect::<Vec<_>>()
            })
    ) {
        let source = Arc::new(SessionTestSource::with_len(16));
        let hnsw_params = HnswParams::new(4, 16)
            .expect("HNSW params must be valid")
            .with_rng_seed(99);

        // Build expected edge set via the direct index.
        let direct_index = CpuHnsw::with_capacity(hnsw_params.clone(), source.len())
            .expect("direct index must allocate");
        let mut expected_edges: Vec<CandidateEdge> = Vec::new();
        for &index in &indices {
            let edges = direct_index
                .insert_harvesting(index, source.as_ref())
                .expect("direct insert must succeed");
            expected_edges.extend(edges);
        }

        // Append the same sequence via the session.
        let mut session = ChutoroBuilder::new()
            .with_hnsw_params(hnsw_params)
            .build_session(Arc::clone(&source))
            .expect("session must build");
        session
            .append(&indices)
            .expect("unique in-bounds append sequence must succeed");

        prop_assert_eq!(
            session.pending_edges,
            expected_edges,
            "pending_edges must match direct harvested edges for the same index sequence"
        );
    }
}
