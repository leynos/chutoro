//! Candidate-bucket and dimension planning for neighbour-scoring benchmarks.

const REALISTIC_BUCKETS: &[usize] = &[8, 16, 24, 32, 48];
const DIAGNOSTIC_BUCKETS: &[usize] = &[256, 1_024];

pub(super) const DIMENSIONS: &[usize] = &[32, 128, 768];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BucketKind {
    Realistic,
    Diagnostic,
}

impl BucketKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Realistic => "realistic",
            Self::Diagnostic => "diagnostic",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CandidateBucket {
    size: usize,
    kind: BucketKind,
}

impl CandidateBucket {
    const fn new(size: usize, kind: BucketKind) -> Self {
        Self { size, kind }
    }

    pub(super) const fn size(self) -> usize {
        self.size
    }

    pub(super) const fn kind_name(self) -> &'static str {
        self.kind.as_str()
    }
}

pub(super) fn all_buckets() -> impl Iterator<Item = CandidateBucket> {
    REALISTIC_BUCKETS
        .iter()
        .copied()
        .map(|size| CandidateBucket::new(size, BucketKind::Realistic))
        .chain(
            DIAGNOSTIC_BUCKETS
                .iter()
                .copied()
                .map(|size| CandidateBucket::new(size, BucketKind::Diagnostic)),
        )
}

pub(super) fn scoring_plan() -> Vec<(usize, CandidateBucket)> {
    DIMENSIONS
        .iter()
        .flat_map(|&dimension| all_buckets().map(move |bucket| (dimension, bucket)))
        .collect()
}

#[cfg(test)]
mod tests {
    //! Unit tests for neighbour-scoring benchmark planning.

    use super::{BucketKind, CandidateBucket, DIMENSIONS, all_buckets, scoring_plan};

    fn occurrence_count(
        plan: &[(usize, CandidateBucket)],
        dimension: usize,
        bucket: CandidateBucket,
    ) -> usize {
        plan.iter()
            .filter(|&&(planned_dimension, planned_bucket)| {
                planned_dimension == dimension && planned_bucket == bucket
            })
            .count()
    }

    #[test]
    fn candidate_bucket_reports_kind_and_size() {
        let bucket = CandidateBucket::new(8, BucketKind::Realistic);

        assert_eq!(bucket.kind_name(), "realistic");
        assert_eq!(bucket.size(), 8);
    }

    #[test]
    fn scoring_plan_covers_each_dimension_and_bucket_once() {
        let plan = scoring_plan();
        let buckets = all_buckets().collect::<Vec<_>>();

        assert_eq!(plan.len(), DIMENSIONS.len() * buckets.len());
        for &dimension in DIMENSIONS {
            for &bucket in &buckets {
                assert_eq!(occurrence_count(&plan, dimension, bucket), 1);
            }
        }
    }
}
