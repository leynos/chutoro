//! Attachment validation outcomes shared by graph constructors.

/// Reasons a node context fails validation during attachment.
///
/// Shared by the production and Kani constructors so both map the same checks
/// to their own error representations.
#[derive(Clone, Copy, Debug)]
pub(super) enum AttachNodeError {
    /// The node's requested level exceeds the configured maximum.
    LevelExceedsMax,
    /// The node identifier lies outside the graph's preallocated slots.
    OutsideCapacity,
    /// The node identifier already occupies a graph slot.
    Duplicate,
}
