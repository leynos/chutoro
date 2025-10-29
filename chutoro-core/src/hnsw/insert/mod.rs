//! HNSW insertion workflow.
//!
//! Provides the planning and application phases for inserting nodes into the HNSW
//! graph. Planning computes the descent path and layer-by-layer neighbour
//! candidates without holding write locks. Application mutates the graph
//! structure, performs bidirectional linking, and schedules trimming jobs for
//! nodes that exceed maximum connection limits.

mod executor;
mod planner;

pub(super) use executor::{InsertionExecutor, TrimJob, TrimResult};
pub(super) use planner::{InsertionPlanner, PlanningInputs};
