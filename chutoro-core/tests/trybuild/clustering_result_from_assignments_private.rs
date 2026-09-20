//! Compile-fail fixture proving the panicking constructor is crate-private.

use chutoro_core::{ClusterId, ClusteringResult};

fn main() {
    let _ = ClusteringResult::from_assignments(vec![ClusterId::new(0)]);
}
