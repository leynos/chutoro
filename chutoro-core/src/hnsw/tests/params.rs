//! Parameter validation tests for HNSW configuration structures.

use crate::hnsw::HnswParams;

#[test]
fn accepts_equal_search_and_connection_width() {
    let params = HnswParams::new(8, 8).expect("equal widths must be valid");
    assert_eq!(params.max_connections(), 8);
    assert_eq!(params.ef_construction(), 8);
}
