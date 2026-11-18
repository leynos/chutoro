//! Compile-fail fixture: `HierarchyErrorCode` cannot be matched exhaustively.

use chutoro_core::HierarchyErrorCode;

fn describe(code: HierarchyErrorCode) -> &'static str {
    match code {
        HierarchyErrorCode::EmptyDataset => "empty dataset",
        HierarchyErrorCode::MinClusterSizeTooLarge => "minimum cluster size too large",
        HierarchyErrorCode::InvalidEdgeWeight => "invalid edge weight",
        HierarchyErrorCode::InvalidEdgeEndpoint => "invalid edge endpoint",
        HierarchyErrorCode::InvalidForestReference => "invalid forest reference",
        HierarchyErrorCode::InvalidClusterReference => "invalid cluster reference",
        HierarchyErrorCode::InvalidPointReference => "invalid point reference",
    }
}

fn main() {
    let _ = describe(HierarchyErrorCode::EmptyDataset);
}