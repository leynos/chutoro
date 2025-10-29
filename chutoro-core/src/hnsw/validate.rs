use super::{
    distance_cache::{DistanceCache, LookupOutcome},
    error::HnswError,
};
use crate::DataSource;

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
        let metric = source.metric_descriptor();
        let mut results: Vec<Option<f32>> = vec![None; candidates.len()];
        let mut pending = Vec::new();
        for (idx, &candidate) in candidates.iter().enumerate() {
            match cache.begin_lookup(&metric, query, candidate) {
                LookupOutcome::Hit(value) => results[idx] = Some(value),
                LookupOutcome::Miss(miss) => pending.push((idx, miss)),
            }
        }
        if !pending.is_empty() {
            let missing: Vec<usize> = pending.iter().map(|(idx, _)| candidates[*idx]).collect();
            let computed = source.batch_distances(query, &missing)?;
            for ((idx, miss), value) in pending.into_iter().zip(computed.into_iter()) {
                let value = cache.complete_miss(miss, value)?;
                results[idx] = Some(value);
            }
        }
        let resolved: Option<Vec<f32>> = results.into_iter().collect();
        Ok(resolved.expect("distance cache resolved all candidates"))
    } else {
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
}
