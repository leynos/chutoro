//! Lean graph constructors used by Kani proof harnesses.
//!
//! These wrappers avoid constructing production errors with formatted strings,
//! which would otherwise inflate CBMC's symbolic formula before an invariant is
//! evaluated.

use super::NodeContext;
use super::core::{AttachNodeError, Graph};
use crate::hnsw::types::EntryPoint;

impl AttachNodeError {
    /// Returns the static reason used by the Kani constructors.
    const fn static_reason(self) -> &'static str {
        match self {
            Self::LevelExceedsMax => "node level exceeds max_level",
            Self::OutsideCapacity => "node is outside pre-allocated capacity",
            Self::Duplicate => "node already exists",
        }
    }
}

/// Reports whether `level` should replace the current entry level.
fn should_promote_entry(current: Option<EntryPoint>, level: usize) -> bool {
    level > current.map_or(0, |entry| entry.level)
}

impl Graph {
    /// Inserts the first Kani node without constructing formatted errors.
    pub(crate) fn insert_first_for_kani(&mut self, ctx: NodeContext) -> Result<(), &'static str> {
        self.attach_node_for_kani(ctx)?;
        self.promote_entry_to(ctx);
        Ok(())
    }

    /// Attaches a Kani node without constructing formatted production errors.
    pub(crate) fn attach_node_for_kani(&mut self, ctx: NodeContext) -> Result<(), &'static str> {
        self.attach_node_inner(ctx)
            .map_err(AttachNodeError::static_reason)
    }

    /// Exposes entry-promotion criteria to Kani proofs.
    pub(crate) fn should_promote_entry_for_kani(current: Option<EntryPoint>, level: usize) -> bool {
        should_promote_entry(current, level)
    }
}
