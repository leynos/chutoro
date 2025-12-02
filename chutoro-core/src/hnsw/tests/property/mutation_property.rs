//! Stateful mutation property covering add/delete/reconfigure sequences.
//!
//! Builds an index from a sampled fixture, executes a series of randomized
//! mutations, and revalidates every structural invariant after each step. The
//! operations are derived from [`MutationPlan`] so the shrinking process can
//! replay the failing sequence deterministically.

use proptest::{
    prop_assume,
    test_runner::{TestCaseError, TestCaseResult},
};
use tracing::debug;

use super::{
    support::DenseVectorSource,
    types::{HnswFixture, HnswParamsSeed, MutationOperationSeed, MutationPlan},
};
use crate::{CpuHnsw, DataSource, HnswError, HnswParams};

const MIN_MUTATION_FIXTURE_LEN: usize = 2;
const MAX_MUTATION_FIXTURE_LEN: usize = 48;

pub(super) fn run_mutation_property(fixture: HnswFixture, plan: MutationPlan) -> TestCaseResult {
    let (mut active_params, source, len) = setup_mutation_index(&fixture)?;
    let mut index = CpuHnsw::with_capacity(active_params.clone(), len)
        .map_err(|err| TestCaseError::fail(format!("with_capacity(len={len}) failed: {err}")))?;
    let mut pools = MutationPools::new(len);
    let mut ctx = MutationContext::new(&mut index, &mut pools, &source);

    bootstrap_and_validate(&mut ctx, plan.initial_population_hint, len)?;

    let applied = apply_mutation_steps(&mut ctx, &mut active_params, &plan)?;

    prop_assume!(applied > 0);
    Ok(())
}

struct MutationContext<'a> {
    index: &'a mut CpuHnsw,
    pools: &'a mut MutationPools,
    source: &'a DenseVectorSource,
}

impl<'a> MutationContext<'a> {
    fn new(
        index: &'a mut CpuHnsw,
        pools: &'a mut MutationPools,
        source: &'a DenseVectorSource,
    ) -> Self {
        Self {
            index,
            pools,
            source,
        }
    }
}

fn setup_mutation_index(
    fixture: &HnswFixture,
) -> Result<(HnswParams, DenseVectorSource, usize), TestCaseError> {
    let params = fixture
        .params
        .build()
        .map_err(|err| TestCaseError::fail(format!("invalid params: {err}")))?;
    let source = fixture
        .clone()
        .into_source()
        .map_err(|err| TestCaseError::fail(format!("fixture -> source failed: {err}")))?;
    let len = source.len();
    prop_assume!(len >= MIN_MUTATION_FIXTURE_LEN);
    prop_assume!(len <= MAX_MUTATION_FIXTURE_LEN);
    Ok((params, source, len))
}

fn bootstrap_and_validate(
    ctx: &mut MutationContext<'_>,
    initial_population_hint: u16,
    len: usize,
) -> Result<(), TestCaseError> {
    let initial_population = derive_initial_population(initial_population_hint, len);
    let seeded = ctx.pools.bootstrap(initial_population);
    for node in seeded {
        ctx.index.insert(node, ctx.source).map_err(|err| {
            TestCaseError::fail(format!("initial insert node {node} failed: {err}"))
        })?;
    }

    #[cfg(test)]
    heal_graph(ctx.index);

    ctx.index.invariants().check_all().map_err(|err| {
        debug!(len, %err, "invariants failed after bootstrap");
        TestCaseError::fail(format!(
            "invariants failed after bootstrap (len={len}): {err}"
        ))
    })?;
    debug!(len, "invariants passed after bootstrap");
    Ok(())
}

fn apply_mutation_steps(
    ctx: &mut MutationContext<'_>,
    active_params: &mut HnswParams,
    plan: &MutationPlan,
) -> Result<usize, TestCaseError> {
    let mut applied = 0_usize;
    for (step, operation) in plan.operations.iter().enumerate() {
        let outcome = {
            let mut ctx = MutationRunner {
                index: ctx.index,
                pools: ctx.pools,
                source: ctx.source,
                active_params,
            };
            ctx.apply(operation)?
        };
        if outcome.applied {
            applied += 1;
        }
        debug!(
            step,
            applied = outcome.applied,
            summary = %outcome.summary,
            "hnsw mutation step"
        );
        #[cfg(test)]
        heal_graph(ctx.index);
        ctx.index.invariants().check_all().map_err(|err| {
            debug!(
                step,
                applied,
                len = ctx.index.len(),
                operation = ?operation,
                %err,
                "invariants failed after mutation step"
            );
            TestCaseError::fail(format!(
                "invariants failed after step {step} ({:?}): {err}",
                operation,
            ))
        })?;
        debug!(
            step,
            applied,
            len = ctx.index.len(),
            operation = ?operation,
            "invariants passed after mutation step"
        );
    }
    Ok(applied)
}

#[cfg(test)]
fn heal_graph(index: &CpuHnsw) {
    index.heal_for_test();
}

struct MutationRunner<'ctx> {
    index: &'ctx mut CpuHnsw,
    pools: &'ctx mut MutationPools,
    source: &'ctx DenseVectorSource,
    active_params: &'ctx mut HnswParams,
}

impl<'ctx> MutationRunner<'ctx> {
    fn apply(
        &mut self,
        operation: &MutationOperationSeed,
    ) -> Result<OperationOutcome, TestCaseError> {
        match operation {
            MutationOperationSeed::Add { slot_hint } => {
                if let Some(node) = self.pools.select_available(*slot_hint) {
                    let insert_result = self.index.insert(node, self.source);
                    insert_result.map_err(|err| mutation_fail("insert", node, err))?;
                    self.pools.mark_inserted(node);
                    Ok(OperationOutcome::applied(format!("add node {node}")))
                } else {
                    Ok(OperationOutcome::skipped("add skipped: no available nodes"))
                }
            }
            MutationOperationSeed::Delete { slot_hint } => {
                let Some(node) = self.pools.select_inserted(*slot_hint) else {
                    return Ok(OperationOutcome::skipped(
                        "delete skipped: no inserted nodes",
                    ));
                };
                let delete_result = self.index.delete_node_for_test(node);
                match delete_result {
                    Ok(true) => {
                        self.pools.mark_deleted(node);
                        Ok(OperationOutcome::applied(format!("delete node {node}")))
                    }
                    Ok(false) => Ok(OperationOutcome::skipped(format!(
                        "delete skipped: node {node} missing"
                    ))),
                    Err(err @ HnswError::GraphInvariantViolation { .. }) => {
                        Ok(OperationOutcome::skipped(format!("delete aborted: {err}")))
                    }
                    Err(err) => Err(mutation_fail("delete", node, err)),
                }
            }
            MutationOperationSeed::Reconfigure { params } => {
                let next = next_reconfigure_params(self.active_params, params)?;
                self.index.reconfigure_for_test(next.clone());
                let summary = format!(
                    "reconfigure -> M={} ef={}",
                    next.max_connections(),
                    next.ef_construction(),
                );
                *self.active_params = next;
                Ok(OperationOutcome::applied(summary))
            }
        }
    }
}

pub(super) fn derive_initial_population(hint: u16, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let base = usize::from(hint) % len;
    let upper_bound = len / 2;
    base.max(1).min(upper_bound.max(1))
}

fn next_reconfigure_params(
    current: &HnswParams,
    seed: &HnswParamsSeed,
) -> Result<HnswParams, TestCaseError> {
    let max_connections = seed.max_connections.max(current.max_connections());
    let ef_construction = seed
        .ef_construction
        .max(max_connections)
        .max(current.ef_construction());
    let mut params = HnswParams::new(max_connections, ef_construction)
        .map_err(|err| TestCaseError::fail(format!("reconfigure parameters invalid: {err}")))?;
    params = params
        .with_level_multiplier(seed.level_multiplier)
        .with_max_level(seed.max_level)
        .with_rng_seed(seed.rng_seed)
        .with_distance_cache_config(*current.distance_cache_config());
    Ok(params)
}

fn mutation_fail(action: &str, node: usize, err: HnswError) -> TestCaseError {
    TestCaseError::fail(format!("{action} node {node} failed: {err}"))
}

#[derive(Default)]
struct MutationPools {
    inserted: Vec<usize>,
    available: Vec<usize>,
}

impl MutationPools {
    fn new(capacity: usize) -> Self {
        Self {
            inserted: Vec::new(),
            available: (0..capacity).collect(),
        }
    }

    fn bootstrap(&mut self, count: usize) -> Vec<usize> {
        let take = count.min(self.available.len());
        let seeded: Vec<usize> = self.available.drain(0..take).collect();
        self.inserted.extend(&seeded);
        seeded
    }

    fn select_available(&self, hint: u16) -> Option<usize> {
        Self::select_from(&self.available, hint)
    }

    fn select_inserted(&self, hint: u16) -> Option<usize> {
        Self::select_from(&self.inserted, hint)
    }

    fn mark_inserted(&mut self, node: usize) {
        if Self::remove_value(&mut self.available, node) {
            self.inserted.push(node);
        }
    }

    fn mark_deleted(&mut self, node: usize) {
        if Self::remove_value(&mut self.inserted, node) {
            self.available.push(node);
            self.available.sort_unstable();
        }
    }

    fn select_from(list: &[usize], hint: u16) -> Option<usize> {
        if list.is_empty() {
            None
        } else {
            let idx = usize::from(hint) % list.len();
            list.get(idx).copied()
        }
    }

    fn remove_value(list: &mut Vec<usize>, value: usize) -> bool {
        if let Some(position) = list.iter().position(|&candidate| candidate == value) {
            list.remove(position);
            true
        } else {
            false
        }
    }
}

struct OperationOutcome {
    applied: bool,
    summary: String,
}

impl OperationOutcome {
    fn applied(summary: impl Into<String>) -> Self {
        Self {
            applied: true,
            summary: summary.into(),
        }
    }

    fn skipped(summary: impl Into<String>) -> Self {
        Self {
            applied: false,
            summary: summary.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn derive_initial_population_respects_bounds() {
        assert_eq!(derive_initial_population(0, 4), 1);
        assert_eq!(derive_initial_population(5, 4), 1);
        assert_eq!(derive_initial_population(6, 7), 3);
    }

    #[rstest]
    fn derive_initial_population_edge_cases() {
        assert_eq!(derive_initial_population(0, 0), 0);
        assert_eq!(derive_initial_population(1, 0), 0);
        assert_eq!(derive_initial_population(10, 0), 0);

        for hint in [0_u16, 1, 4, 10, u16::MAX] {
            assert_eq!(
                derive_initial_population(hint, 1),
                1,
                "expected 1 initial element for len=1 and hint={hint}",
            );
        }
    }

    #[rstest]
    fn mutation_pools_track_membership() {
        let mut pools = MutationPools::new(4);
        let seeded = pools.bootstrap(2);
        assert_eq!(seeded, vec![0, 1]);
        assert_eq!(pools.select_available(0), Some(2));
        assert_eq!(pools.select_inserted(0), Some(0));
        pools.mark_deleted(0);
        assert_eq!(pools.select_available(0), Some(0));
        pools.mark_inserted(2);
        assert!(pools.select_available(0).is_some());
    }

    #[rstest]
    fn reconfigure_never_decreases_max_connections() {
        let current = HnswParams::new(8, 32).expect("params");
        let seed = HnswParamsSeed {
            max_connections: 2,
            ef_construction: 4,
            level_multiplier: 0.5,
            max_level: 6,
            rng_seed: 11,
        };
        let params = next_reconfigure_params(&current, &seed).expect("reconfigure");
        assert_eq!(params.max_connections(), 8);
        assert!(params.ef_construction() >= params.max_connections());
    }
}
