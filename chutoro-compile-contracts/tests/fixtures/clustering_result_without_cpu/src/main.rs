//! Compile-fail fixture proving the CPU-only constructor is unavailable.

use chutoro_core::{ClusterId, ClusteringResult};

fn main() {
    let _ = ClusteringResult::from_assignments(vec![ClusterId::new(0)]);
}
