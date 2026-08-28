//! Error types for single-linkage hierarchy construction.
//!
//! The hierarchy builder exposes a compact semantic error surface while its
//! forest and condensation phases remain implementation details.

/// Errors returned by hierarchy extraction.
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum HierarchyError {
    /// Hierarchy extraction requires at least one node.
    #[error("cannot extract a hierarchy for an empty dataset")]
    EmptyDataset,
    /// The configured minimum cluster size exceeds the dataset size.
    #[error("min_cluster_size {min_cluster_size} exceeds node_count {node_count}")]
    MinClusterSizeTooLarge {
        /// Number of points in the dataset.
        node_count: usize,
        /// Minimum cluster size requested by the caller.
        min_cluster_size: usize,
    },
    /// An MST edge weight was invalid for hierarchy extraction.
    #[error("invalid MST edge weight {weight} for edge ({left}, {right})")]
    InvalidEdgeWeight {
        /// Endpoint id for the offending edge.
        left: usize,
        /// Other endpoint id for the offending edge.
        right: usize,
        /// Invalid weight value observed on the edge.
        weight: f32,
    },
    /// An MST edge referenced an endpoint outside the dataset.
    #[error("MST edge endpoint {endpoint} is outside dataset size {node_count}")]
    InvalidEdgeEndpoint {
        /// Endpoint identifier that did not belong to the dataset.
        endpoint: usize,
        /// Number of nodes in the dataset.
        node_count: usize,
    },
    /// The constructed linkage forest referenced a missing node.
    #[error("linkage forest references missing node {node_id}")]
    InvalidForestReference {
        /// Identifier of the missing linkage node.
        node_id: usize,
    },
    /// The condensation process referenced a missing cluster.
    #[error("condensation references missing cluster {cluster_id}")]
    InvalidClusterReference {
        /// Identifier of the missing cluster.
        cluster_id: usize,
    },
    /// The condensation process referenced a point outside the dataset.
    #[error("condensation references point {point_id} outside dataset size {node_count}")]
    InvalidPointReference {
        /// Identifier of the missing dataset point.
        point_id: usize,
        /// Number of points in the dataset.
        node_count: usize,
    },
}

impl HierarchyError {
    /// Returns a stable, machine-readable error code for the variant.
    #[must_use]
    pub const fn code(&self) -> HierarchyErrorCode {
        match self {
            Self::EmptyDataset => HierarchyErrorCode::EmptyDataset,
            Self::MinClusterSizeTooLarge { .. } => HierarchyErrorCode::MinClusterSizeTooLarge,
            Self::InvalidEdgeWeight { .. } => HierarchyErrorCode::InvalidEdgeWeight,
            Self::InvalidEdgeEndpoint { .. } => HierarchyErrorCode::InvalidEdgeEndpoint,
            Self::InvalidForestReference { .. } => HierarchyErrorCode::InvalidForestReference,
            Self::InvalidClusterReference { .. } => HierarchyErrorCode::InvalidClusterReference,
            Self::InvalidPointReference { .. } => HierarchyErrorCode::InvalidPointReference,
        }
    }
}

/// Machine-readable error codes for [`HierarchyError`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HierarchyErrorCode {
    /// The caller requested hierarchy extraction for an empty dataset.
    EmptyDataset,
    /// The configured minimum cluster size exceeds the dataset size.
    MinClusterSizeTooLarge,
    /// An input edge weight was invalid for hierarchy extraction.
    InvalidEdgeWeight,
    /// An input edge endpoint was outside the dataset.
    InvalidEdgeEndpoint,
    /// A constructed linkage forest referenced a missing node.
    InvalidForestReference,
    /// Condensation referenced a missing cluster.
    InvalidClusterReference,
    /// Condensation referenced a point outside the dataset.
    InvalidPointReference,
}

impl HierarchyErrorCode {
    /// Returns the symbolic identifier for logging and metrics surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyDataset => "EMPTY_DATASET",
            Self::MinClusterSizeTooLarge => "MIN_CLUSTER_SIZE_TOO_LARGE",
            Self::InvalidEdgeWeight => "INVALID_EDGE_WEIGHT",
            Self::InvalidEdgeEndpoint => "INVALID_EDGE_ENDPOINT",
            Self::InvalidForestReference => "INVALID_FOREST_REFERENCE",
            Self::InvalidClusterReference => "INVALID_CLUSTER_REFERENCE",
            Self::InvalidPointReference => "INVALID_POINT_REFERENCE",
        }
    }
}
