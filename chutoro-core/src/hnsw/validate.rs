use super::{
    distance_cache::{DistanceCache, LookupOutcome, PendingMiss},
    error::HnswError,
};
use crate::{DataSource, MetricDescriptor};

fn lookup_or_compute<D: DataSource + Sync>(
    cache: Option<&DistanceCache>,
    source: &D,
    left: usize,
    right: usize,
) -> Result<f32, HnswError> {
    if let Some(cache) = cache {
        let metric = source.metric_descriptor();
        match cache.begin_lookup(&metric, left, right) {
            LookupOutcome::Hit(value) => Ok(value),
            LookupOutcome::Miss(pending) => {
                let value = source.distance(left, right)?;
                cache.complete_miss(pending, value)
            }
        }
    } else {
        Ok(source.distance(left, right)?)
    }
}

fn batch_lookup_or_compute<D: DataSource + Sync>(
    cache: &DistanceCache,
    source: &D,
    query: usize,
    candidates: &[usize],
) -> Result<Vec<f32>, HnswError> {
    let context = CacheBatch::new(cache, source, query, candidates);
    let mut results: Vec<Option<f32>> = vec![None; candidates.len()];
    let mut pending = Vec::new();

    context.populate(&mut results, &mut pending);

    if !pending.is_empty() {
        context.resolve(pending, &mut results)?;
    }

    Ok(results
        .into_iter()
        .map(|value| value.expect("cache lookups resolve every candidate"))
        .collect())
}

pub(crate) fn validate_distance<D: DataSource + Sync>(
    cache: Option<&DistanceCache>,
    source: &D,
    left: usize,
    right: usize,
) -> Result<f32, HnswError> {
    let value = lookup_or_compute(cache, source, left, right)?;
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
        batch_lookup_or_compute(cache, source, query, candidates)
    } else {
        validate_batch_without_cache(source, query, candidates)
    }
}

fn validate_batch_without_cache<D: DataSource + Sync>(
    source: &D,
    query: usize,
    candidates: &[usize],
) -> Result<Vec<f32>, HnswError> {
    let distances = source.batch_distances(query, candidates)?;
    for (&candidate, &distance) in candidates.iter().zip(distances.iter()) {
        if !distance.is_finite() {
            return Err(HnswError::NonFiniteDistance {
                left: query,
                right: candidate,
            });
        }
    }
    Ok(distances)
}

struct CacheBatch<'a, D: DataSource + Sync> {
    cache: &'a DistanceCache,
    source: &'a D,
    query: usize,
    candidates: &'a [usize],
    metric: MetricDescriptor,
}

impl<'a, D: DataSource + Sync> CacheBatch<'a, D> {
    fn new(cache: &'a DistanceCache, source: &'a D, query: usize, candidates: &'a [usize]) -> Self {
        Self {
            cache,
            source,
            query,
            candidates,
            metric: source.metric_descriptor(),
        }
    }

    fn populate(&self, results: &mut [Option<f32>], pending: &mut Vec<(usize, PendingMiss)>) {
        for (index, &candidate) in self.candidates.iter().enumerate() {
            match self.cache.begin_lookup(&self.metric, self.query, candidate) {
                LookupOutcome::Hit(value) => results[index] = Some(value),
                LookupOutcome::Miss(miss) => pending.push((index, miss)),
            }
        }
    }

    fn resolve(
        &self,
        pending: Vec<(usize, PendingMiss)>,
        results: &mut [Option<f32>],
    ) -> Result<(), HnswError> {
        let missing: Vec<usize> = pending
            .iter()
            .map(|(index, _)| self.candidates[*index])
            .collect();
        let computed = self.source.batch_distances(self.query, &missing)?;

        for ((index, miss), value) in pending.into_iter().zip(computed.into_iter()) {
            let value = self.cache.complete_miss(miss, value)?;
            results[index] = Some(value);
        }

        Ok(())
    }
}
