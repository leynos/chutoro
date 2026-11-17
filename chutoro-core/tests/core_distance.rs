//! Availability fixture for the public core-distance helper.
//!
//! The helper is part of the `cpu` feature surface, so this binary is empty
//! when that feature is off and, when it is on, proves that the crate-root
//! re-export exists with the documented signature. Behaviour is covered by the
//! unit and doctests beside the implementation.
#![cfg(feature = "cpu")]

use std::num::NonZeroUsize;

use chutoro_core::{Neighbour, core_distance_from_neighbours};

#[test]
fn helper_is_exported_at_the_crate_root() {
    let neighbours = [
        Neighbour {
            id: 1,
            distance: 0.25,
        },
        Neighbour {
            id: 2,
            distance: 0.75,
        },
    ];
    let min_cluster_size = NonZeroUsize::new(2).expect("two is non-zero");

    assert_eq!(
        core_distance_from_neighbours(&neighbours, min_cluster_size),
        0.75
    );
}
