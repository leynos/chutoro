//! Shared helper utilities used by the CPU HNSW implementation to keep
//! `cpu.rs` focused on graph orchestration while isolating search maintenance
//! logic for reuse and easier testing.

use std::num::NonZeroUsize;

use crate::DataSource;

use super::{
    distance_cache::{DistanceCache, LookupOutcome},
    error::HnswError,
    types::Neighbour,
    validate::{validate_batch_distances, validate_distance},
};

/// Bundles the context required to ensure a search result includes the query
/// item when enough capacity is available.
pub(crate) struct EnsureQueryArgs<'a, D: DataSource + Sync> {
    pub source: &'a D,
    pub query: usize,
    pub ef: NonZeroUsize,
    pub neighbours: &'a mut Vec<Neighbour>,
}

/// Orders neighbours by ascending distance and, when distances tie, by the
/// neighbour identifier to guarantee deterministic ordering.
pub(crate) fn normalise_neighbour_order(neighbours: &mut [Neighbour]) {
    neighbours.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.id.cmp(&right.id))
    });
}

/// Ensures the query node is included within the `ef` window when the current
/// neighbour list has spare capacity.
///
/// # Examples
/// ```rust,ignore
/// # use std::num::NonZeroUsize;
/// # use chutoro_core::hnsw::{helpers::{EnsureQueryArgs, ensure_query_present, normalise_neighbour_order}, distance_cache::DistanceCache};
/// # use chutoro_core::{DistanceCacheConfig, datasource::DataSource};
/// # struct Dummy;
/// # impl DataSource for Dummy {
/// #     fn len(&self) -> usize { 1 }
/// #     fn name(&self) -> &str { "dummy" }
/// #     fn distance(&self, _: usize, _: usize) -> Result<f32, chutoro_core::datasource::DataSourceError> { Ok(0.0) }
/// # }
/// # let cache = DistanceCache::new(DistanceCacheConfig::default());
/// # let mut neighbours = Vec::new();
/// # normalise_neighbour_order(&mut neighbours);
/// ensure_query_present(
///     &cache,
///     EnsureQueryArgs {
///         source: &Dummy,
///         query: 0,
///         ef: NonZeroUsize::new(2).unwrap(),
///         neighbours: &mut neighbours,
///     },
/// )?;
/// assert_eq!(neighbours.len(), 1);
/// # Ok::<(), chutoro_core::hnsw::error::HnswError>(())
/// ```
pub(crate) fn ensure_query_present<D: DataSource + Sync>(
    cache: &DistanceCache,
    args: EnsureQueryArgs<'_, D>,
) -> Result<(), HnswError> {
    let EnsureQueryArgs {
        source,
        query,
        ef,
        neighbours,
    } = args;
    if ef.get() == 1 || neighbours.iter().any(|neighbour| neighbour.id == query) {
        return Ok(());
    }
    let distance = validate_distance(Some(cache), source, query, query)?;
    neighbours.push(Neighbour {
        id: query,
        distance,
    });
    normalise_neighbour_order(neighbours);
    while neighbours.len() > ef.get() {
        neighbours.pop();
    }
    Ok(())
}

/// Computes distances for a candidate batch required by the trim step,
/// returning them in the same order and populating the shared cache.
///
/// # Examples
/// ```rust,ignore
/// # use chutoro_core::{DistanceCacheConfig, datasource::DataSource};
/// # use chutoro_core::hnsw::helpers::batch_distances_for_trim;
/// # use chutoro_core::hnsw::distance_cache::DistanceCache;
/// # struct Dummy(Vec<f32>);
/// # impl DataSource for Dummy {
/// #     fn len(&self) -> usize { self.0.len() }
/// #     fn name(&self) -> &str { "dummy" }
/// #     fn distance(&self, i: usize, j: usize) -> Result<f32, chutoro_core::datasource::DataSourceError> { Ok((self.0[i] - self.0[j]).abs()) }
/// # }
/// let cache = DistanceCache::new(DistanceCacheConfig::default());
/// let source = Dummy(vec![0.0, 1.0, 4.0]);
/// let distances = batch_distances_for_trim(&cache, 0, &[1, 2], &source)?;
/// assert_eq!(distances, vec![1.0, 4.0]);
/// # Ok::<(), chutoro_core::hnsw::error::HnswError>(())
/// ```
pub(crate) fn batch_distances_for_trim<D: DataSource + Sync>(
    cache: &DistanceCache,
    node: usize,
    candidates: &[usize],
    source: &D,
) -> Result<Vec<f32>, HnswError> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let metric = source.metric_descriptor();
    let pending: Vec<_> = candidates
        .iter()
        .map(
            |&candidate| match cache.begin_lookup(&metric, node, candidate) {
                LookupOutcome::Hit(_) => None,
                LookupOutcome::Miss(miss) => Some(miss),
            },
        )
        .collect();
    let distances = validate_batch_distances(None, source, node, candidates)?;
    for (miss, distance) in pending.into_iter().zip(distances.iter()) {
        if let Some(miss) = miss {
            cache.complete_miss(miss, *distance)?;
        }
    }
    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataSourceError, DistanceCacheConfig, datasource::MetricDescriptor};

    fn cache() -> DistanceCache {
        DistanceCache::new(DistanceCacheConfig::default())
    }

    fn run_ensure_query_test(
        initial_neighbours: Vec<Neighbour>,
        ef: usize,
        expected_neighbours: Vec<Neighbour>,
    ) {
        let source = TestSource::new(vec![0.0, 1.0]);
        let mut neighbours = initial_neighbours;
        ensure_query_present(
            &cache(),
            EnsureQueryArgs {
                source: &source,
                query: 0,
                ef: NonZeroUsize::new(ef).expect("ef must be non-zero"),
                neighbours: &mut neighbours,
            },
        )
        .expect("ensure_query_present must succeed");
        assert_eq!(neighbours, expected_neighbours);
    }

    #[test]
    fn ensure_query_added_when_room_available() {
        run_ensure_query_test(
            vec![neighbour(1, 1.0)],
            2,
            vec![neighbour(0, 0.0), neighbour(1, 1.0)],
        );
    }

    #[test]
    fn ensure_query_skips_when_capacity_is_one() {
        run_ensure_query_test(vec![neighbour(1, 1.0)], 1, vec![neighbour(1, 1.0)]);
    }

    #[test]
    fn ensure_query_noop_when_present() {
        run_ensure_query_test(vec![neighbour(0, 0.0)], 1, vec![neighbour(0, 0.0)]);
    }

    #[test]
    fn batch_distances_populates_cache() {
        let cache = cache();
        let source = TestSource::new(vec![0.0, 1.0, 4.0]);
        let distances = batch_distances_for_trim(&cache, 0, &[1, 2], &source)
            .expect("batch distances must succeed");
        assert_eq!(distances, vec![1.0, 4.0]);

        let metric = source.metric_descriptor();
        matches!(
            cache.begin_lookup(&metric, 0, 1),
            LookupOutcome::Hit(value) if (value - 1.0).abs() < f32::EPSILON
        )
        .then_some(())
        .expect("distance cache must record lookup");
    }

    fn neighbour(id: usize, distance: f32) -> Neighbour {
        Neighbour { id, distance }
    }

    #[derive(Clone)]
    struct TestSource {
        data: Vec<f32>,
    }

    impl TestSource {
        fn new(data: Vec<f32>) -> Self {
            Self { data }
        }
    }

    impl DataSource for TestSource {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn name(&self) -> &str {
            "test"
        }

        fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
            Ok((self.data[left] - self.data[right]).abs())
        }

        fn metric_descriptor(&self) -> MetricDescriptor {
            MetricDescriptor::new("test")
        }
    }
}
