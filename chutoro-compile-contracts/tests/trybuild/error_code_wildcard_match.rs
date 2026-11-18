//! Compile-pass fixture: a wildcard arm accepts future error-code variants.

use chutoro_core::{HierarchyErrorCode, HnswErrorCode, MstErrorCode};

fn describe_mst(code: MstErrorCode) -> &'static str {
    match code {
        MstErrorCode::EmptyGraph => "empty graph",
        _ => "other MST failure",
    }
}

fn describe_hierarchy(code: HierarchyErrorCode) -> &'static str {
    match code {
        HierarchyErrorCode::EmptyDataset => "empty dataset",
        _ => "other hierarchy failure",
    }
}

fn describe_hnsw(code: HnswErrorCode) -> &'static str {
    match code {
        HnswErrorCode::EmptyBuild => "empty build",
        _ => "other HNSW failure",
    }
}

fn main() {
    let _ = describe_mst(MstErrorCode::EmptyGraph);
    let _ = describe_hierarchy(HierarchyErrorCode::EmptyDataset);
    let _ = describe_hnsw(HnswErrorCode::EmptyBuild);
}