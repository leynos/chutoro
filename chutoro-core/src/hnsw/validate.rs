use super::{
    distance_cache::{DistanceCache, LookupOutcome, PendingMiss},
    error::HnswError,
};
use crate::{DataSource, MetricDescriptor};

pub(crate) fn validate_distance<D: DataSource + Sync>(
    cache: Option<&DistanceCache>,
    source: &D,
    left: usize,
    right: usize,
) -> Result<f32, HnswError> {
    let metric = source.metric_descriptor();
    let value = if let Some(cache) = cache {
        match cache.begin_lookup(&metric, left, right) {
            LookupOutcome::Hit(value) => value,
            LookupOutcome::Miss(pending) => {
                let value = source.distance(left, right)?;
                cache.complete_miss(pending, value)?
            }
        }
    } else {
        source.distance(left, right)?
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HnswError::NonFiniteDistance { left, right })
    }
}

pub(crate) fn validate_batch_distances<D: DataSource + Sync>(
    cache: Option<&DistanceCache>,
    source: &D,
    query: usize,
    candidates: &[usize],
) -> Result<Vec<f32>, HnswError> {
    if let Some(cache) = cache {
        validate_batch_with_cache(cache, source, query, candidates)
    } else {
        validate_batch_without_cache(source, query, candidates)
    }
}

fn validate_batch_with_cache<D: DataSource + Sync>(
    cache: &DistanceCache,
    source: &D,
    query: usize,
    candidates: &[usize],
) -> Result<Vec<f32>, HnswError> {
    let context = CacheBatchContext::new(cache, source, query, candidates);
    let mut results: Vec<Option<f32>> = vec![None; candidates.len()];
    let mut pending = Vec::new();

    populate_cached_results(&context, &mut results, &mut pending);

    if !pending.is_empty() {
        resolve_pending_distances(&context, pending, &mut results)?;
    }

    let resolved: Option<Vec<f32>> = results.into_iter().collect();
    Ok(resolved.expect("distance cache resolved all candidates"))
}

fn populate_cached_results<D: DataSource + Sync>(
    context: &CacheBatchContext<'_, D>,
    results: &mut [Option<f32>],
    pending: &mut Vec<(usize, PendingMiss)>,
) {
    for (idx, &candidate) in context.candidates.iter().enumerate() {
        match context
            .cache
            .begin_lookup(&context.metric, context.query, candidate)
        {
            LookupOutcome::Hit(value) => results[idx] = Some(value),
            LookupOutcome::Miss(miss) => pending.push((idx, miss)),
        }
    }
}

fn resolve_pending_distances<D: DataSource + Sync>(
    context: &CacheBatchContext<'_, D>,
    pending: Vec<(usize, PendingMiss)>,
    results: &mut [Option<f32>],
) -> Result<(), HnswError> {
    let missing: Vec<usize> = pending
        .iter()
        .map(|(idx, _)| context.candidates[*idx])
        .collect();
    let computed = context.source.batch_distances(context.query, &missing)?;

    for ((idx, miss), value) in pending.into_iter().zip(computed.into_iter()) {
        let value = context.cache.complete_miss(miss, value)?;
        results[idx] = Some(value);
    }

    Ok(())
}

fn validate_batch_without_cache<D: DataSource + Sync>(
    source: &D,
    query: usize,
    candidates: &[usize],
) -> Result<Vec<f32>, HnswError> {
    let distances = source.batch_distances(query, candidates)?;
    for (candidate, distance) in candidates.iter().copied().zip(distances.iter().copied()) {
        if !distance.is_finite() {
            return Err(HnswError::NonFiniteDistance {
                left: query,
                right: candidate,
            });
        }
    }
    Ok(distances)
}

struct CacheBatchContext<'a, D: DataSource + Sync> {
    cache: &'a DistanceCache,
    source: &'a D,
    metric: MetricDescriptor,
    query: usize,
    candidates: &'a [usize],
}

impl<'a, D: DataSource + Sync> CacheBatchContext<'a, D> {
    fn new(cache: &'a DistanceCache, source: &'a D, query: usize, candidates: &'a [usize]) -> Self {
        Self {
            cache,
            source,
            metric: source.metric_descriptor(),
            query,
            candidates,
        }
    }
}
