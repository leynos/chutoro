//! End-to-end coverage for excess-of-mass stability ties.
//!
//! Integer-valued metrics, such as the Levenshtein distance used by the text
//! provider, routinely give a parent cluster exactly the same stability as the
//! sum of its children. The design selects the parent only when its stability
//! strictly exceeds the children's, so a tie must keep the separate groups.

use chutoro_core::{ChutoroBuilder, DataSource, DataSourceError};

/// Levenshtein distances between `apple`, `apply`, `apples`, `appel`,
/// `banana`, `bananas`, `bandana`, and `banane`, in that order.
const LEVENSHTEIN: [[u8; 8]; 8] = [
    [0, 1, 1, 2, 5, 6, 6, 4],
    [1, 0, 2, 2, 5, 6, 6, 5],
    [1, 2, 0, 2, 6, 5, 6, 5],
    [2, 2, 2, 0, 5, 6, 6, 5],
    [5, 5, 6, 5, 0, 1, 1, 1],
    [6, 6, 5, 6, 1, 0, 2, 2],
    [6, 6, 6, 6, 1, 2, 0, 2],
    [4, 5, 5, 5, 1, 2, 2, 0],
];

/// A pre-computed integer distance matrix exposed as a data source.
struct WordDistances;

impl DataSource for WordDistances {
    fn len(&self) -> usize {
        LEVENSHTEIN.len()
    }

    fn name(&self) -> &'static str {
        "word-families"
    }

    fn distance(&self, i: usize, j: usize) -> Result<f32, DataSourceError> {
        let row = LEVENSHTEIN
            .get(i)
            .ok_or(DataSourceError::OutOfBounds { index: i })?;
        let value = row
            .get(j)
            .ok_or(DataSourceError::OutOfBounds { index: j })?;
        Ok(f32::from(*value))
    }
}

#[test]
fn integer_distance_ties_keep_word_families_apart() {
    let chutoro = ChutoroBuilder::new()
        .with_min_cluster_size(2)
        .build()
        .expect("a minimum cluster size of two is valid");

    let result = chutoro
        .run(&WordDistances)
        .expect("the word-family matrix should cluster");

    let labels: Vec<u64> = result.assignments().iter().map(|id| id.get()).collect();
    assert_eq!(result.cluster_count(), 2);
    assert_eq!(labels, vec![0, 0, 0, 0, 1, 1, 1, 1]);
}
