//! Distance validation helpers for HNSW operations.
//!
//! This module exports `validate_distance`, `validate_batch_distances`, and
//! `validate_batch_without_cache` for checked single and batched distance
//! lookups. Their shared `lookup_or_compute` helper consults an optional
//! `DistanceCache` before falling back to `DataSource::distance` using the
//! source's `metric_descriptor`, bridging `distance_cache.rs` cache state with
//! `error.rs` failure reporting through `HnswError`.

use super::{
    distance_cache::{DistanceCache, LookupOutcome, PendingMiss},
    error::HnswError,
};
use crate::{DataSource, MetricDescriptor};

/// Return one cached or newly computed finite distance.
fn lookup_or_compute<D: DataSource + Sync>(
    cache_option: Option<&DistanceCache>,
    source: &D,
    left: usize,
    right: usize,
) -> Result<f32, HnswError> {
    if let Some(cache) = cache_option {
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

/// Resolve a batch through the cache before computing its misses.
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

    ensure_all_resolved(query, candidates, results)
}

/// Validate one distance from an optional cache-backed lookup.
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

/// Validate a batch of query-to-candidate distances.
pub(crate) fn validate_batch_distances<D: DataSource + Sync>(
    cache_option: Option<&DistanceCache>,
    source: &D,
    query: usize,
    candidates: &[usize],
) -> Result<Vec<f32>, HnswError> {
    cache_option.map_or_else(
        || validate_batch_without_cache(source, query, candidates),
        |cache| batch_lookup_or_compute(cache, source, query, candidates),
    )
}

/// Validate source-provided batch distances when caching is unavailable.
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

/// Inputs shared while resolving a cache-backed batch lookup.
struct CacheBatch<'a, D: DataSource + Sync> {
    /// Cache that owns lookup and miss-completion state.
    cache: &'a DistanceCache,
    /// Source used to compute uncached distances.
    source: &'a D,
    /// Query node shared by every candidate.
    query: usize,
    /// Candidate nodes ordered with the result slots.
    candidates: &'a [usize],
    /// Source metric attached to cache keys.
    metric: MetricDescriptor,
}

impl<'a, D: DataSource + Sync> CacheBatch<'a, D> {
    /// Capture cache, source, query, candidates, and their metric descriptor.
    fn new(cache: &'a DistanceCache, source: &'a D, query: usize, candidates: &'a [usize]) -> Self {
        Self {
            cache,
            source,
            query,
            candidates,
            metric: source.metric_descriptor(),
        }
    }

    /// Fill hit result slots and collect cache misses with their indices.
    fn populate(&self, results: &mut [Option<f32>], pending: &mut Vec<(usize, PendingMiss)>) {
        for (index, (&candidate, result)) in
            self.candidates.iter().zip(results.iter_mut()).enumerate()
        {
            match self.cache.begin_lookup(&self.metric, self.query, candidate) {
                LookupOutcome::Hit(value) => *result = Some(value),
                LookupOutcome::Miss(miss) => pending.push((index, miss)),
            }
        }
    }

    /// Compute, validate, and store each pending cache miss.
    fn resolve(
        &self,
        pending: Vec<(usize, PendingMiss)>,
        results: &mut [Option<f32>],
    ) -> Result<(), HnswError> {
        let missing: Vec<usize> = pending
            .iter()
            .map(|(index, _)| {
                self.candidates.get(*index).copied().ok_or_else(|| {
                    HnswError::GraphInvariantViolation {
                        message: format!(
                            "cached batch validation: missing candidate at result index {index}",
                        ),
                    }
                })
            })
            .collect::<Result<_, _>>()?;
        let computed = self.source.batch_distances(self.query, &missing)?;

        if computed.len() != pending.len() {
            return Err(HnswError::InvalidParameters {
                reason: format!(
                    "data source returned {} distance(s) for {} pending candidates during cached batch validation",
                    computed.len(),
                    pending.len()
                ),
            });
        }

        for ((index, miss), computed_value) in pending.into_iter().zip(computed.into_iter()) {
            let cached_value = self.cache.complete_miss(miss, computed_value)?;
            let result =
                results
                    .get_mut(index)
                    .ok_or_else(|| HnswError::GraphInvariantViolation {
                        message: format!(
                            "cached batch validation: missing result slot at index {index}",
                        ),
                    })?;
            *result = Some(cached_value);
        }

        Ok(())
    }
}

/// Convert resolved slots into distances or report the first unresolved candidate.
fn ensure_all_resolved(
    query: usize,
    candidates: &[usize],
    results: Vec<Option<f32>>,
) -> Result<Vec<f32>, HnswError> {
    if results.len() != candidates.len() {
        return Err(HnswError::InvalidParameters {
            reason: format!(
                "distance cache returned {} result slot(s) for {} candidates during batch validation",
                results.len(),
                candidates.len()
            ),
        });
    }

    let mut resolved = Vec::with_capacity(results.len());
    for (candidate, result_slot) in candidates.iter().zip(results.into_iter()) {
        match result_slot {
            Some(distance) => resolved.push(distance),
            None => {
                return Err(HnswError::InvalidParameters {
                    reason: format!(
                        "distance cache left candidate {candidate} unresolved for query {query}"
                    ),
                });
            }
        }
    }

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    //! Unit tests for distance validation.

    use super::*;

    #[test]
    fn ensure_all_resolved_reports_unresolved_candidate() {
        let err = ensure_all_resolved(0, &[1, 2], vec![Some(0.1), None])
            .expect_err("unresolved candidate must be reported");

        match err {
            HnswError::InvalidParameters { reason } => {
                assert!(
                    reason.contains("candidate 2"),
                    "reason must describe the unresolved candidate: {reason}"
                );
            }
            other => panic!("expected invalid parameters error, got {other:?}"),
        }
    }
}
