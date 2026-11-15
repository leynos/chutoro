//! Shared helper utilities used by the CPU HNSW implementation to keep
//! `cpu.rs` focused on graph orchestration while isolating search maintenance
//! logic for reuse and easier testing.

use std::num::NonZeroUsize;

use crate::{DataSource, MetricDescriptor};

use super::{
    distance_cache::{DistanceCache, LookupOutcome, PendingMiss},
    error::HnswError,
    types::Neighbour,
    validate::{validate_batch_distances, validate_distance},
};

/// Bundles the context required to ensure a search result includes the query
/// item when enough capacity is available.
pub(crate) struct EnsureQueryArgs<'a, D: DataSource + Sync> {
    /// Data source used to evaluate a missing query candidate.
    pub source: &'a D,
    /// Query identifier that may need to be inserted.
    pub query: usize,
    /// Search-result capacity that bounds query insertion.
    pub ef: NonZeroUsize,
    /// Mutable result list updated when the query is absent.
    pub neighbours: &'a mut Vec<Neighbour>,
}

/// Orders neighbours by ascending distance and, when distances tie, by the
/// neighbour identifier to guarantee deterministic ordering.
pub(crate) fn normalize_neighbour_order(neighbours: &mut [Neighbour]) {
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
/// use chutoro_core::{DistanceCacheConfig, DataSource, DataSourceError};
/// use crate::datasource::MetricDescriptor;
/// use crate::hnsw::distance_cache::DistanceCache;
/// use crate::hnsw::helpers::{
///     EnsureQueryArgs, ensure_query_present, normalize_neighbour_order,
/// };
///
/// struct Dummy;
///
/// impl DataSource for Dummy {
///     fn len(&self) -> usize { 1 }
///     fn name(&self) -> &str { "dummy" }
///     fn distance(&self, _: usize, _: usize) -> Result<f32, DataSourceError> { Ok(0.0) }
///     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("dummy") }
/// }
///
/// let cache = DistanceCache::new(DistanceCacheConfig::default());
/// let mut neighbours = Vec::new();
/// normalize_neighbour_order(&mut neighbours);
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
/// # Ok::<(), crate::hnsw::error::HnswError>(())
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
    normalize_neighbour_order(neighbours);
    neighbours.truncate(ef.get());
    Ok(())
}

/// Computes distances for a candidate batch required by the trim step,
/// returning them in the same order and populating the shared cache.
///
/// # Examples
/// ```rust,ignore
/// use chutoro_core::{DistanceCacheConfig, DataSource, DataSourceError};
/// use crate::datasource::MetricDescriptor;
/// use crate::hnsw::distance_cache::DistanceCache;
/// use crate::hnsw::helpers::batch_distances_for_trim;
///
/// struct Dummy(Vec<f32>);
///
/// impl DataSource for Dummy {
///     fn len(&self) -> usize { self.0.len() }
///     fn name(&self) -> &str { "dummy" }
///     fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
///         Ok((self.0[i] - self.0[j]).abs())
///     }
///     fn metric_descriptor(&self) -> MetricDescriptor { MetricDescriptor::new("dummy") }
/// }
///
/// let cache = DistanceCache::new(DistanceCacheConfig::default());
/// let source = Dummy(vec![0.0, 1.0, 4.0]);
/// let distances = batch_distances_for_trim(&cache, 0, &[1, 2], &source)?;
/// assert_eq!(distances, vec![1.0, 4.0]);
/// # Ok::<(), crate::hnsw::error::HnswError>(())
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
    let query = TrimQuery {
        cache,
        metric: &metric,
        node,
    };
    let mut distances = vec![0.0; candidates.len()];
    let lookups = fill_cached_trim_distances(&query, candidates, &mut distances);
    if lookups.miss_candidates.is_empty() {
        return Ok(distances);
    }

    let miss_distances = validate_batch_distances(None, source, node, &lookups.miss_candidates)?;
    if miss_distances.len() != lookups.miss_candidates.len() {
        return Err(HnswError::InvalidParameters {
            reason: format!(
                "miss distance count ({}) mismatches candidates ({})",
                miss_distances.len(),
                lookups.miss_candidates.len()
            ),
        });
    }

    apply_trim_miss_distances(cache, &mut distances, lookups.miss_meta, miss_distances)?;
    Ok(distances)
}

/// The fixed context of one trim distance batch.
///
/// The three fields travel together through every lookup in the batch: they
/// identify which cache entry a candidate distance belongs to. Grouping them
/// keeps the helpers below within the repository's argument limit without
/// splitting a context that is only meaningful whole.
struct TrimQuery<'a> {
    /// Cache shared with the rest of the graph maintenance.
    cache: &'a DistanceCache,
    /// Metric the distance is measured in, part of the cache key.
    metric: &'a MetricDescriptor,
    /// Node whose candidate distances are being resolved.
    node: usize,
}

/// Cache hits and outstanding misses collected from a trim distance batch.
struct TrimLookups {
    /// Candidates whose distance was not cached, in candidate order.
    miss_candidates: Vec<usize>,
    /// Pending miss handles paired with the result slot each one fills.
    miss_meta: Vec<(usize, PendingMiss)>,
}

/// Fills the cached distances and reports which candidates still need a query.
///
/// Separated from [`batch_distances_for_trim`] because the cache probe is one
/// decision — reuse the cached value or record a miss — while applying the
/// computed misses is a different one, with its own error path.
fn fill_cached_trim_distances(
    query: &TrimQuery<'_>,
    candidates: &[usize],
    distances: &mut [f32],
) -> TrimLookups {
    let mut miss_candidates = Vec::new();
    let mut miss_meta = Vec::new();

    for (index, (&candidate, distance_slot)) in candidates.iter().zip(distances).enumerate() {
        match query
            .cache
            .begin_lookup(query.metric, query.node, candidate)
        {
            LookupOutcome::Hit(value) => *distance_slot = value,
            LookupOutcome::Miss(miss) => {
                miss_candidates.push(candidate);
                miss_meta.push((index, miss));
            }
        }
    }

    TrimLookups {
        miss_candidates,
        miss_meta,
    }
}

/// Stores freshly computed miss distances in the cache and the result batch.
///
/// The miss handles are completed before the slot is written, so a cache
/// rejection leaves the distance buffer untouched.
fn apply_trim_miss_distances(
    cache: &DistanceCache,
    distances: &mut [f32],
    miss_meta: Vec<(usize, PendingMiss)>,
    miss_distances: Vec<f32>,
) -> Result<(), HnswError> {
    for ((index, miss), distance) in miss_meta.into_iter().zip(miss_distances) {
        cache.complete_miss(miss, distance)?;
        *distances
            .get_mut(index)
            .ok_or_else(|| HnswError::InvalidParameters {
                reason: format!("distance slot {index} is outside the candidate batch"),
            })? = distance;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Unit tests for HNSW helper routines.

    use rstest::{fixture, rstest};

    use super::*;
    use crate::{DataSourceError, DistanceCacheConfig, datasource::MetricDescriptor};

    #[fixture]
    fn cache() -> DistanceCache {
        DistanceCache::new(DistanceCacheConfig::default())
    }

    /// Runs [`ensure_query_present`] over `initial_neighbours` and returns the
    /// resulting list.
    ///
    /// A fallible query rather than an assertion helper, so the calling test
    /// owns both the failure diagnostics and the comparison.
    fn ensured_neighbours(
        cache: &DistanceCache,
        initial_neighbours: Vec<Neighbour>,
        ef: usize,
    ) -> Result<Vec<Neighbour>, HnswError> {
        let source = TestSource::new(vec![0.0, 1.0]);
        let mut neighbours = initial_neighbours;
        let search_ef = NonZeroUsize::new(ef).ok_or_else(|| HnswError::InvalidParameters {
            reason: "ef must be non-zero".to_owned(),
        })?;
        ensure_query_present(
            cache,
            EnsureQueryArgs {
                source: &source,
                query: 0,
                ef: search_ef,
                neighbours: &mut neighbours,
            },
        )?;
        Ok(neighbours)
    }

    #[rstest]
    #[case::added_when_room_available(
        vec![neighbour(1, 1.0)],
        2,
        vec![neighbour(0, 0.0), neighbour(1, 1.0)]
    )]
    #[case::skipped_when_capacity_is_one(vec![neighbour(1, 1.0)], 1, vec![neighbour(1, 1.0)])]
    #[case::noop_when_query_present(vec![neighbour(0, 0.0)], 1, vec![neighbour(0, 0.0)])]
    fn ensure_query_present_matches_expected_window(
        cache: DistanceCache,
        #[case] initial_neighbours: Vec<Neighbour>,
        #[case] ef: usize,
        #[case] expected_neighbours: Vec<Neighbour>,
    ) {
        let neighbours = ensured_neighbours(&cache, initial_neighbours, ef)
            .expect("ensure_query_present must succeed");
        assert_eq!(neighbours, expected_neighbours);
    }

    #[rstest]
    fn ensure_query_evicts_furthest_when_full(cache: DistanceCache) {
        let mut neighbours = vec![neighbour(1, 1.0), neighbour(2, 2.0)];
        let source = TestSource::new(vec![0.0, 1.0, 2.0]);
        ensure_query_present(
            &cache,
            EnsureQueryArgs {
                source: &source,
                query: 0,
                ef: NonZeroUsize::new(2).expect("ef must be non-zero"),
                neighbours: &mut neighbours,
            },
        )
        .expect("ensure_query_present must succeed");
        assert_eq!(neighbours.len(), 2);
        assert!(neighbours.iter().any(|neighbour| neighbour.id == 0));
        assert!(neighbours.iter().any(|neighbour| neighbour.id == 1));
        assert!(neighbours.iter().all(|neighbour| neighbour.id != 2));
    }

    #[rstest]
    fn batch_distances_populates_cache(cache: DistanceCache) {
        let source = TestSource::new(vec![0.0, 1.0, 4.0]);
        let distances = batch_distances_for_trim(&cache, 0, &[1, 2], &source)
            .expect("batch distances must succeed");
        assert_eq!(distances, vec![1.0, 4.0]);

        let metric = source.metric_descriptor();
        assert!(matches!(
            cache.begin_lookup(&metric, 0, 1),
            LookupOutcome::Hit(value) if value.total_cmp(&1.0).is_eq()
        ));
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

        fn name(&self) -> &'static str {
            "test"
        }

        fn distance(&self, left: usize, right: usize) -> Result<f32, DataSourceError> {
            let left_value = self
                .data
                .get(left)
                .ok_or(DataSourceError::OutOfBounds { index: left })?;
            let right_value = self
                .data
                .get(right)
                .ok_or(DataSourceError::OutOfBounds { index: right })?;
            Ok(left_value
                .mul_add(1.0, std::ops::Neg::neg(*right_value))
                .abs())
        }

        fn metric_descriptor(&self) -> MetricDescriptor {
            MetricDescriptor::new("test")
        }
    }
}
