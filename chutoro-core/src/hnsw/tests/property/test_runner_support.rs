//! Shared proptest runner and coverage-budget helpers for HNSW property tests.

mod budget_selection;
mod runner_wrappers;

pub(super) use budget_selection::{
    JobKind, ShrinkIterations, StackSize, TestCases, idempotency_cases, idempotency_shrink_iters,
    mutation_cases, mutation_shrink_iters, search_cases, search_shrink_iters,
    select_idempotency_cases, select_idempotency_shrink_iters, select_mutation_cases,
    select_mutation_shrink_iters, select_search_cases, select_search_shrink_iters,
};
pub(super) use runner_wrappers::{run_idempotency_test, run_mutation_test, run_search_test};
