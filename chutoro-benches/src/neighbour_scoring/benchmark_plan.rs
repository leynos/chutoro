//! Candidate-bucket and dimension planning for neighbour-scoring benchmarks.

/// Candidate counts representing expected production workloads.
const REALISTIC_BUCKETS: &[usize] = &[8, 16, 24, 32, 48];
/// Candidate counts used to expose large-workload behaviour.
const DIAGNOSTIC_BUCKETS: &[usize] = &[256, 1_024];

/// Vector dimensions covered by the scoring benchmark matrix.
pub(super) const DIMENSIONS: &[usize] = &[32, 128, 768];

/// Intended workload class for a candidate-count bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BucketKind {
    /// Expected production-sized candidate count.
    Realistic,
    /// Larger candidate count used for diagnostics.
    Diagnostic,
}

impl BucketKind {
    /// Return the stable label for this workload class.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Realistic => "realistic",
            Self::Diagnostic => "diagnostic",
        }
    }
}

/// Candidate-count bucket paired with its workload class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CandidateBucket {
    /// Number of candidate points in the bucket.
    size: usize,
    /// Workload class represented by the bucket.
    kind: BucketKind,
}

impl CandidateBucket {
    /// Construct one candidate bucket.
    const fn new(size: usize, kind: BucketKind) -> Self {
        Self { size, kind }
    }

    /// Return the number of candidates in this bucket.
    pub(super) const fn size(self) -> usize {
        self.size
    }

    /// Return the stable workload-class label for this bucket.
    pub(super) const fn kind_name(self) -> &'static str {
        self.kind.as_str()
    }
}

/// Iterate over every planned candidate bucket.
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

/// Build the Cartesian product of dimensions and candidate buckets.
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
