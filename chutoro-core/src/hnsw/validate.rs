use super::error::HnswError;
use crate::DataSource;

pub(crate) fn validate_distance<D: DataSource + Sync>(
    source: &D,
    left: usize,
    right: usize,
) -> Result<f32, HnswError> {
    let value = source.distance(left, right)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HnswError::NonFiniteDistance { left, right })
    }
}

pub(crate) fn validate_batch_distances<D: DataSource + Sync>(
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
