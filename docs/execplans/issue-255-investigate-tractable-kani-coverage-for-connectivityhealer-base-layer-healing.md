# Investigate tractable Kani coverage for `ConnectivityHealer` base-layer healing

This ExecPlan (execution plan) is a living document. The sections `Constraints`,
`Tolerances`, `Risks`, `Progress`, `Surprises & discoveries`, `Decision log`,
`Outcomes & retrospective`, `Conformance basis`, and `Verification plan` must
be kept up to date as work proceeds.

Status: IN PROGRESS

Branch:
`issue-255-investigate-tractable-kani-coverage-for-connectivityhealer-base-layer-healing`

Issue: [chutoro#255](https://github.com/leynos/chutoro/issues/255). Backlink:
[PR #227 comment 5583341792](https://github.com/leynos/chutoro/pull/227#issuecomment-5583341792).

## Purpose / big picture

`ConnectivityHealer::ensure_base_connectivity` is the routine that repairs the
base layer (HNSW level 0) after insertion reconciliation has isolated a node.
When node `n` is disconnected, the routine links the graph entry node `e` to
`n` and `n` back to `e`, then drains a work queue of any node displaced by an
eviction. The behaviour is currently covered by ordinary unit tests, which run
the production `VisitedSet = HashSet<usize>` implementation. Kani builds swap
that alias for `LinearVisitedSet`, a bounded linear-scan substitute, because
`HashSet`'s randomized SipHash state is symbolic under Kani. The substitute's
insertion semantics are pinned by `rstest` equivalence cases; the healing path
itself is **not** reached by any Kani harness.

A previous attempt to drive `ensure_base_connectivity` directly with a two-node
harness timed out in symbolic execution: it aborted 2,590 paths after twenty
minutes. The investigation recorded in
`docs/kani-full-hnsw-hypothesis-testing.md` concluded that the healer path was
past the tractable CBMC (C Bounded Model Checker) state space. This plan
re-opens that conclusion with a narrower, fully concrete harness and a
measured, one-lever-at-a-time experiment. The deliverable is either:

1. a passing, non-vacuous `#[kani::proof]` that calls the unchanged production
   `ensure_base_connectivity` and asserts that an isolated node and the entry
   node acquire reciprocal base-layer links; or
2. a documented measurement of why no bounded design is tractable, retaining
   the existing unit tests and `LinearVisitedSet` equivalence tests as the
   substitute.

Observable success for outcome 1 is `make kani` (or `make kani-full` if the
harness fits only the nightly budget) printing `VERIFICATION:- SUCCESSFUL` for
the new harness and `Complete - N successfully verified harnesses, 0 failures`
for its tier, with `RUSTFLAGS="-D warnings"` applied. Observable success for
outcome 2 is a dated `Findings to Date` entry in
`docs/adrs/adr-0002-adoption-of-kani-formal-verification.md` and a measured
limits section in `docs/kani-full-hnsw-hypothesis-testing.md` naming the
commands, exit codes, elapsed times, and path counts.

## Constraints

- The production semantics of `ConnectivityHealer::ensure_base_connectivity`,
  `link_new_node_inner`, and the `VisitedSet` selection must not change.
- The pull-request Kani gate (`make kani` and `.github/workflows/kani-pr.yml`)
  must not be weakened. A new heavy harness must not be added to the fast tier.
- The existing normal healing tests and the `LinearVisitedSet` semantic
  equivalence tests must remain.
- The timed-out direct two-node harness must not be added to PR #227.
- `tests/workflow_contracts/kani_fast_tier_test.py` must keep passing without
  modification.
- No broad `allow`/`expect` attributes may be introduced to silence warnings.
- Every new module starts with a `//!` comment; public items carry Rustdoc.
- Files stay under 400 lines. Comments use en-GB-oxendict spelling.
- Validation must include `RUSTFLAGS="-D warnings" make kani` and
  `RUSTFLAGS="-D warnings" make kani-full`. If the unchanged baseline already
  fails under that assignment, record the pre-existing failure in the ADR
  rather than weakening the gate or hiding the warning.
- Commit after each coherent change, gate each commit with the repository
  targets, and keep the working tree clean of probe code that is not promoted.

## Tolerances (exception triggers)

- Time: a single Kani harness run that exceeds 20 minutes of wall-clock time is
  intractable for this investigation. Stop that configuration, record the
  measured bounds, and move to the next lever or to outcome 2.
- Levers: at most one tractability lever (solver, unwind bound, removal of a
  panic-capable setup step, one bounded symbolic choice) changes per
  measurement, so each effect is attributable.
- Code size: the promoted harness plus the Kani bridge should stay under 150 net
  lines of production/test code. If it needs more, stop and review the design.
- Documentation size: ADR and hypothesis-testing updates under 120 net lines.
  ExecPlan files are excluded from every size count.
- Iterations: if the concrete rung still times out after all documented levers,
  stop experimenting and write outcome 2. Do not widen bounds to compensate.
- Interface: a public API change is out of scope; the bridge must be
  `#[cfg(kani)] pub(crate)`.
- Architecture: replacing production `HashSet` with a different production
  structure, or extracting the healer into a model with different semantics, is
  out of scope and requires escalation.

## Risks

- Risk: the previous 2,590-path abort is reproduced even with all-concrete
  inputs, implying an inherent CBMC cost in `Vec<Vec<usize>>` mutation.
  Severity: high. Likelihood: medium. Mitigation: measure a minimal
  `Vec<Vec<usize>>`-only probe to separate graph-storage cost from healer cost,
  then record the limit if no lever helps.
- Risk: `RUSTFLAGS="-D warnings"` fails on the unchanged baseline under
  `cfg(kani)` because proof-only helpers are partly unused. Severity: medium.
  Likelihood: high. Mitigation: record the baseline failure verbatim in the ADR
  and CI evidence; do not add broad allows.
- Risk: `make kani-full` is too long to iterate against. Severity: low.
  Likelihood: medium. Mitigation: measure harnesses individually first; run the
  full tier once for the baseline and once for the final integration.
- Risk: a harness passes vacuously because the setup silently fails or the
  assertion path is unreachable. Severity: high. Likelihood: medium.
  Mitigation: assert setup preconditions, use `kani::cover!`, and run two
  temporary mutations (skip the reciprocal link; pre-seed the edge) to prove
  the harness rejects them.
- Risk: added Kani harness changes the fast-tier workflow contract. Severity:
  low. Likelihood: low. Mitigation: run `make test-workflow-contracts` and
  inspect the Makefile-derived harness list.

## Progress

- [x] (2026-11-21 10:05Z) Branch renamed and remote tracking established;
  Lody session renamed.
- [x] (2026-11-21 10:10Z) Reconnaissance: read `connectivity.rs`,
  `kani_proofs/`, `insert/mod.rs`, `providers/params/graph` constructors,
  Makefile Kani targets, ADR 0002, and the hypothesis-testing document.
- [x] (2026-11-21 10:12Z) `make kani-check` passed: Kani 0.67.0 matches the
  pin.
- [x] (2026-11-21 10:28Z) Baseline `make kani` completed: 6 of 6 harnesses
  successful, 1001.5 s wall, exit 0. Per-harness times are recorded under
  `Artefacts and notes`. `make kani-full` and the `RUSTFLAGS="-D warnings"`
  variants are part of the final gate run.
- [x] (2026-11-21 10:30Z) Probe: the all-concrete two-node healing harness
  passes in 9.57 s verification (31.0 s wall including codegen), 0 of 1240
  checks failed, 2 of 2 covers satisfied, and **zero** aborted paths. The
  recorded 2,590-path abort is not inherent to the healing path.
- [x] (2026-11-21 10:29Z) First run failed only on a `memcmp` unwinding
  assertion at `unwind(4)`; `unwind(9)` covers the eight-byte slice comparison
  and passes. No other lever was needed.
- [x] (2026-11-21 10:30Z) `EP-M1` bounded harness with exact postconditions
  and `kani::cover!` passes; `make check-fmt` is clean and `make lint` is
  running.
- [x] (2026-11-21 10:38Z) Negative control 1 (skip the reverse directed link)
  fails with `healing must link the isolated node back to the entry node`;
  negative control 2 (pre-seed `0 <-> 1`) fails with
  `healing precondition: both base lists must start empty`. Both mutations
  reverted.
- [x] (2026-11-21 10:50Z) Widened rung (three nodes plus one symbolic boolean,
  `unwind(17)`) exceeded 600 s of solver time with more than 6,400 aborted
  paths and was stopped. This probe changed the node count, the symbolic
  choice, and the unwind bound together, and it stopped before the 20-minute
  limit, so it does not establish a general boundary for symbolic widening.
  Widened harness removed.
- [x] `EP-M2` complete: the concrete rung is non-vacuous. The one widened probe
  tried was stopped past 600 s, which was not enough evidence to retain it.
- [x] (2026-11-21 11:00Z) `EP-M3` tier integration: the harness command is in
  the `kani:` fast-tier recipe at `--default-unwind 9`, and the developers'
  guide, ADR 0002, hypothesis-testing document, contents index, and ExecPlan
  record the outcome.
- [x] (2026-11-21 12:46Z) `make check-fmt`, `make typecheck`,
  `make markdownlint`, and `make nixie` pass. `make lint` reached clean rustdoc
  and Clippy stages in one run; Whitaker was cut off by that background task's
  limit and is re-run by the repository stop hook.
- [x] (2026-11-21 13:58Z) `RUSTFLAGS="-D warnings" make kani` passed: 7 of 7
  harnesses, 2262.5 s wall, exit 0; the healing harness verified in 8.42 s with
  both covers satisfied.
- [x] (2026-11-21 14:42Z) `RUSTFLAGS="-D warnings" make kani-full` passed:
  18 of 18 `chutoro-core` harnesses and both `chutoro-providers-dense`
  harnesses, 2523.6 s wall, exit 0; the healing harness verified in 7.71 s with
  both covers satisfied.
- [x] (2026-11-21 13:14Z) Commit `b6e8780` created with the code, Makefile, and
  documentation changes.
- [x] (2026-11-21 16:43Z) `make test-workflow-contracts` passed: 851 tests, one
  skip, exit 0 (38m33s of pytest time, itself delayed by a nested Cargo compile
  in the admission queue).
- [x] (2026-11-21 16:36Z) The compile-contract test that timed out at 1800 s in
  the first `make test` run passed standalone in 685 s once its fixture was
  warm, confirming the timeout was environmental. The full suite is being rerun
  with warm caches.
- [x] (2026-11-21 19:02Z) Canonical `make test` passed: 1523 tests, zero
  failures, one skip, and all doctests, in 16m14s. An earlier attempt hit the
  suite's 80-minute global timeout while the host's compile-admission queue was
  saturated; the same compile-contract fixtures passed once warm.
- [x] (2026-11-21 17:41Z) Draft pull request
  [#443](https://github.com/leynos/chutoro/pull/443) opened with the issue,
  Lody session, and phase links.
- [ ] CodeRabbit review and any follow-up findings.

## Surprises & discoveries

- Observation: Kani 0.67.0 and the pinned toolchain are already installed and
  `make kani-check` matches `tools/kani/VERSION`. Evidence: `make kani-check`
  printed `Kani 0.67.0 matches from ...`. Impact: no provisioning step is
  needed.
- Observation: the host reports 24 CPUs and 125 GiB of memory, larger than the
  six-core development machine cited in the 2026-08-24 investigation. Evidence:
  `nproc` and `free -g`. Impact: timings here are not directly comparable with
  the older 20-minute observations; record both the configuration and the wall
  time.
- Observation: compile admission is active (`RUSTC_WRAPPER`, four slots) and
  healthy with zero waiting requests. Evidence: `build-limits-status`. Impact:
  Kani builds may queue; a slow start is not a proof failure.
- Observation: the all-concrete two-node healer harness aborted **zero** paths
  and verified in 9.57 s. The earlier recorded failure aborted 2,590 paths over
  twenty minutes. Evidence: `/tmp/kani-healer-unwind9-...out` contains no
  `aborting path on assume(false)` lines. Impact: the intractability was not
  inherent to `ensure_base_connectivity`; a concrete fixture is well inside
  budget. The concrete run has no nondeterminism and never reconstructed the
  earlier harness, so its zero abort count is consistent with, but does not
  prove, symbolic branching in that harness.

## Decision log

- D1. Decision: run the production `ensure_base_connectivity` through a
  `#[cfg(kani)]` bridge in `chutoro-core/src/hnsw/insert/mod.rs`, mirroring
  `ensure_reverse_edge_for_kani`, rather than duplicating the routine in the
  harness. Rationale: the acceptance criterion requires reaching connectivity
  healing, and a duplicated model would prove nothing about production code. A
  bridge adds visibility only under `cfg(kani)`. Date/Author: 2026-11-21,
  implementing agent.
- D2. Decision: target a fully concrete two-node fixture first.
  Rationale: a deterministic path is a single symbolic execution path; if it is
  still intractable, the cost is in the library model, not in input branching,
  and that distinction determines whether symbolic widening is worth trying.
  Date/Author: 2026-11-21, implementing agent.
- D3. Decision: keep `RUSTFLAGS="-D warnings"` evidence separate from the
  functional proof result. Rationale: the nightly workflow explicitly opts out
  of warning denial because proof-only helpers can be partly unused under
  `cfg(kani)`; a pre-existing `dead_code` failure must be reported, not fixed
  with a broad allow. Date/Author: 2026-11-21, implementing agent.
- D4. Decision: retain only the concrete two-node rung; delete the three-node
  symbolic sibling. Rationale: the concrete rung verifies in 9.57 s with zero
  aborted paths, while the sibling exceeded 600 s with more than 6,400 aborted
  paths. Keeping a harness that fails the budget would weaken the gate or force
  a nightly-only proof with poor value per minute. Date/Author: 2026-11-21,
  implementing agent.
- D5. Decision: add the harness to the `kani:` fast-tier recipe. Rationale: it
  costs about ten seconds of verification, far below the existing 69 to 191 s
  harnesses, and the pull-request gate has a 30-minute timeout. Date/Author:
  2026-11-21, implementing agent.

## Outcomes & retrospective

Outcome 1 was achieved: a bounded, non-vacuous Kani harness reaches the
unchanged production `ConnectivityHealer::ensure_base_connectivity`, asserts
the exact reciprocal base-layer links and the preserved entry point, and runs
in both `make kani` and `make kani-full`.

The observed lesson is that a fully concrete fixture is tractable: the same
production routine on a two-node graph verifies in about ten seconds with zero
aborted paths, whereas the one widened probe tried exceeded ten minutes of
solving. The earlier 2,590-path timeout used a different harness on a different
host, so these runs do not establish its cause. They establish that the
concrete two-node path is not inherently intractable, and that the tested
widening is expensive; the probe is not proof of a general boundary, so further
coverage belongs to tests rather than to wider bounds.

What would be done differently: measure a fully concrete fixture before
concluding that a path is past the CBMC state space. The old conclusion rested
on a single harness whose cost this work did not reproduce.

Residual gaps: eviction cascades, re-healing of evicted nodes, and global
connectivity are not proved. They remain covered by the healing tests, and the
`LinearVisitedSet` equivalence cases remain the substitute's test contract.

## Context and orientation

The healing routine lives in `chutoro-core/src/hnsw/insert/connectivity.rs`
(360 lines). Its public-in-crate entry point is
`ConnectivityHealer::ensure_base_connectivity(node, max_connections)`. The
healer is constructed with `ConnectivityHealer::new(&mut graph)` and mutates
`Graph` adjacency lists through `link_new_node_inner` and `link_directed`,
which use `Node::neighbours_mut`. The `VisitedSet` type alias selects
`HashSet<usize>` in normal and `cfg(test)` builds, so unit tests exercise the
production alias, and `LinearVisitedSet` under `cfg(kani)`. The linear set also
compiles under `cfg(test)` solely so its equivalence tests can run.

The graph is `chutoro-core/src/hnsw/graph`:
`Graph::with_capacity(HnswParams, capacity)`, `insert_first_for_kani`, and
`attach_node_for_kani` build a concrete graph without formatting production
errors. `Node` stores `Vec<Vec<usize>>` adjacency lists by level.
`HnswParams::new_for_kani(1, 1)` yields `max_connections = 1`; the base-layer
connection limit is doubled to 2 by `compute_connection_limit(0, 1)`.

Existing harnesses live in `chutoro-core/src/hnsw/kani_proofs/`. The relevant
patterns are in `bidirectional.rs` and `invariants.rs`: `#[kani::proof]`,
`#[kani::solver(kissat)]`, `#[kani::unwind(4)]`,
`let Ok(…) = … else { kani::assert(false, "…"); return; }`, and a
`# Verification Bounds` Rustdoc section.

Gate commands:

```sh
make kani-check
make kani
make kani-full
make check-fmt
make lint
make test
make test-workflow-contracts
mbake validate Makefile
```

Kani version pin: `tools/kani/VERSION` (0.67.0). Rust pin: 1.99.1. The Kani
library path is supplied by the Makefile as `KANI_ENV`; direct `cargo kani`
invocations need the same `LD_LIBRARY_PATH` or the front end cannot load
`libLLVM`.

Documentation to update: `docs/developers-guide.md` "Kani CI policy",
`docs/adrs/adr-0002-adoption-of-kani-formal-verification.md` "Findings to Date"
and "Change Control", and `docs/kani-full-hnsw-hypothesis-testing.md`.

## Conformance basis

No upstream Terms of Reference or technical design document is specific to this
investigation. The governing artefacts are:

- Issue #255 acceptance criteria (the four bullets quoted in the branch task).
- ADR 0002 (Kani adoption,
  `docs/adrs/adr-0002-adoption-of-kani-formal-verification.md`).
- Developers' guide "Kani CI policy" (`docs/developers-guide.md`).
- `tests/workflow_contracts/kani_fast_tier_test.py`, which fixes the Makefile
  contract for the fast tier.

Derived requirements and gaps:

- `REQ-healer-proof`: a Kani harness must reach
  `ConnectivityHealer::ensure_base_connectivity` and complete within budget.
- `REQ-property`: the harness asserts a meaningful base-layer connectivity
  property.
- `REQ-preserve-tests`: existing healing and equivalence tests stay.
- `REQ-gates`: `RUSTFLAGS="-D warnings" make kani` and `... make kani-full`
  are run and their outcomes recorded.
- `GAP-proof`: no harness currently reaches the healer.

Trace:

```plaintext
REQ-healer-proof -> EP-M1 -> verify_base_layer_healing_links_isolated_node_2_nodes
REQ-property -> EP-M1 -> exact base-list postconditions plus kani::cover!
REQ-preserve-tests -> EP-M3 -> make test / make test-workflow-contracts
REQ-gates -> EP-M3 -> make kani and make kani-full transcripts
```

## Verification plan

### Abstract model

Two level-0 nodes exist: node 0 is the entry point with an empty base list, and
node 1 has an empty base list and is isolated. `ensure_base_connectivity(1, 1)`
runs with the base limit 2. In production semantics, node 0 gains node 1 and
node 1 gains node 0; no eviction occurs because both lists have spare capacity.
The work queue then pops node 1 once more (or is empty, depending on the
`Some(new_node)` branch) and the visited set suppresses reprocessing.

### Invariants and lemmas

- `INV-reciprocal`: after the call, `graph.node(0).neighbours(0) == [1]` and
  `graph.node(1).neighbours(0) == [0]`.
- `INV-entry-preserved`: the entry point still references node 0.
- `INV-non-vacuity`: the fixture really has two level-0 nodes with empty base
  lists before the call; `kani::cover!` conditions are satisfiable.
- `L-link-order`: `link_new_node_inner` inserts the forward link before the
  reverse link and rejects the pair when the reverse link is rejected. This
  lemma explains why a mutation that skips the reverse directed link must fail
  `INV-reciprocal`.

### Axioms (trusted, not verified here)

- `AXIOM-cbmc`: CBMC's treatment of `Vec`/`alloc` and `usize` arithmetic is
  sound for the bounded, concrete operations used here.
- `AXIOM-kani-cfg`: `cargo kani` sets `cfg(kani)` and not `cfg(test)`, so the
  `LinearVisitedSet` and lean constructors selected in this plan are the ones
  compiled.
- `AXIOM-params`: `HnswParams::new_for_kani(1, 1)` is accepted and yields
  `max_connections = 1`, `max_level = 12`.

### Methods per obligation

- Obligation: `INV-reciprocal`.
  Method: bounded model check over a concrete two-node fixture. Rationale: the
  property is about exact post-state of production code; only a path through
  production code can establish it, and the input space is a single concrete
  state (with at most one bounded symbolic choice in a widened rung). Domain:
  two nodes, level 0, base limit 2, fresh lists, entry 0. Artefact:
  `chutoro-core/src/hnsw/kani_proofs/connectivity.rs`. Evidence:
  `cargo kani -p chutoro-core --harness
  verify_base_layer_healing_links_isolated_node_2_nodes`
  prints `VERIFICATION:- SUCCESSFUL`. Non-vacuity: exact pre-state assertions;
  `kani::cover!` after setup and after the postconditions; temporary mutations
  described under `EP-M2`.
- Obligation: `INV-entry-preserved`.
  Method: same harness assertion. Rationale: a healer that changed the entry
  point would make the reciprocal assertion misleading, so the entry is
  asserted separately. Evidence: same harness; failure message identifies the
  entry assertion.
- Obligation: `INV-non-vacuity`.
  Method: `kani::cover!` plus negative controls. Rationale: a passing harness
  with an unreachable assertion proves nothing. Evidence: cover results are
  `SATISFIED`; mutation 1 (skip the reverse directed link) fails the reciprocal
  assertion; mutation 2 (pre-seed `0<->1`) fails the precondition assertion.

### Formal verification decision

Bounded model checking is proportionate: the healer is a small stateful
algorithm with list mutation and queue iteration, and the question is whether a
narrow production path is reachable and correct. A deductive proof (Verus)
would require extracting the algorithm into a pure kernel and would prove a
model, not the production mutation sequence. The residual gap is stated
explicitly: the harness covers a two-node, fresh-list, spare-capacity case, not
eviction cascades, re-healing of isolated queues, or global connectivity.

### Suggested BDD features

Not applicable: the repository has no BDD runner for this surface, and the
observable acceptance evidence is the Kani transcript plus the existing
`rstest` suite. The suggested acceptance scenarios are instead expressed by the
`kani::cover!` witnesses and the two mutations.

## Plan of work

### `EP-M0` baseline and feasibility probe

Measure the unchanged gates, then rebuild the failed probe on a scratch basis
without committing it. The probe calls a `#[cfg(kani)]`
`ensure_base_connectivity_for_kani` bridge with node 1, `max_connections = 1`,
the default solver, and `--default-unwind 10`. Record exit code, elapsed time,
CBMC path count, and whether `core::str` panic-formatting paths appear. If the
probe completes, record the time; if it times out, note the exact abort text.

Go/no-go: a completed concrete probe within budget sends the work to `EP-M1`; a
timeout sends it to the lever measurements or outcome 2.

### `EP-M1` bounded harness

Add
`#[cfg(kani)] pub(crate) fn ensure_base_connectivity_for_kani(graph: &mut
Graph, node: usize, max_connections: usize)`
beside `ensure_reverse_edge_for_kani` in
`chutoro-core/src/hnsw/insert/mod.rs`. It constructs
`connectivity::ConnectivityHealer::new(graph)` and calls the unchanged
`ensure_base_connectivity`. It adds no logic and no `kani::assume`.

Create `chutoro-core/src/hnsw/kani_proofs/connectivity.rs` and register
`mod connectivity;` in `kani_proofs/mod.rs`. The harness:

- builds `HnswParams::new_for_kani(1, 1)`;
- builds `Graph::with_capacity(params, 2)`;
- inserts node 0 with `insert_first_for_kani` and attaches node 1 with
  `attach_node_for_kani`, using `let Ok(...) = ... else { ... }`;
- asserts both nodes expose level 0, node 0 is the entry, and both base lists
  are empty;
- calls `ensure_base_connectivity_for_kani(&mut graph, 1, 1)`;
- asserts node 0's base list is exactly `[1]`, node 1's is exactly `[0]`, and
  the entry is still node 0;
- places `kani::cover!` checks after setup and after the postconditions.

Start with `#[kani::solver(kissat)]` and `#[kani::unwind(4)]`; raise the unwind
only for an unwinding-assertion failure. If the harness times out, change one
lever at a time: solver (`kissat`, `cadical`, `minisat`), unwind, removal of a
remaining panic-capable setup operation.

### `EP-M2` non-vacuity and optional widening

Check every `kani::cover!` result explicitly (`SATISFIED` required;
`UNREACHABLE` does not fail verification). On a scratch branch, temporarily
skip the reverse directed link in `link_new_node_inner`; confirm the harness
fails. Revert. Temporarily pre-seed `0<->1`; confirm the precondition assertion
fails. Revert. Record both.

If the concrete rung passes with headroom, add one bounded symbolic choice at a
time (for example, a third node holding a spare neighbour, selected by
`kani::any::<bool>()`). Keep a widened rung only if it remains within budget.
Add eviction-cascade rungs only if every earlier rung has clear headroom.

### `EP-M3` integration and documentation

If the harness passes and fits the fast-tier budget, add its `cargo kani`
command to the `kani:` recipe and keep `kani-full` discovering it. If it fits
only the nightly budget, leave `kani:` unchanged and confirm the package-wide
sweep runs it. Update the developers' guide tier list, the ADR findings and
change control, and the hypothesis-testing document. If no harness is
tractable, document the measured limits, retain the tests, and reference the
tracking issue.

Then run the full validation sequence and create the draft pull request.

## Concrete steps

Working directory: repository root.

```sh
set -o pipefail
make kani-check 2>&1 | tee /tmp/kani-check-$(git branch --show-current).out
make kani 2>&1 | tee /tmp/kani-baseline-$(git branch --show-current).out
make kani-full 2>&1 | tee /tmp/kani-full-baseline-$(git branch --show-current).out
RUSTFLAGS="-D warnings" make kani 2>&1 | tee /tmp/kani-dwarnings-$(git branch --show-current).out
RUSTFLAGS="-D warnings" make kani-full 2>&1 | tee /tmp/kani-full-dwarnings-$(git branch --show-current).out
```

A single harness, matching the Makefile environment:

```sh
LD_LIBRARY_PATH="$HOME/.kani/kani-0.67.0/toolchain/lib:$LD_LIBRARY_PATH" \
  timeout 1200s cargo kani -p chutoro-core --default-unwind 10 \
  --harness verify_base_layer_healing_links_isolated_node_2_nodes \
  2>&1 | tee /tmp/kani-healer-$(git branch --show-current).out
```

Final gates, in order:

```sh
make check-fmt
make lint
make test
make test-workflow-contracts
mbake validate Makefile
```

## Validation and acceptance

Outcome 1 is accepted when the new harness prints `VERIFICATION:- SUCCESSFUL`,
all `kani::cover!` results are `SATISFIED`, both negative controls fail for the
intended reason, `make kani` (or `kani-full`) reports zero failures, and the
final gates above pass. Outcome 2 is accepted when the ADR and hypothesis
document record the environment, baseline, each lever, path counts, and the
decision to retain the test-based substitute, and the same final gates pass.

## Idempotence and recovery

Kani runs are read-only against the working tree. Scratch mutations must be
made on a scratch branch or reverted with `git checkout -- <path>` before any
commit. Logs live in `/tmp` and are safe to overwrite. If a Kani run is
interrupted, re-run it; compile admission queues the work without damaging
other agents' builds.

## Artefacts and notes

### `EP-M0` baseline: unchanged `make kani`

Environment: Kani 0.67.0 (`tools/kani/VERSION`), rustc 1.99.1, 24 CPUs, 125 GiB
RAM, compile admission healthy with four slots.

Log: `/tmp/kani-baseline-issue-255-...out` (full path recorded on the branch).

| Harness                                                       | Verification time | Result     |
| ------------------------------------------------------------- | ----------------- | ---------- |
| `verify_bidirectional_links_smoke_2_nodes_1_layer`            | 69.2 s            | SUCCESSFUL |
| `verify_bidirectional_links_reconciliation_2_nodes_1_layer`   | 149.6 s           | SUCCESSFUL |
| `verify_mst_structural_correctness_4_nodes`                   | 191.4 s           | SUCCESSFUL |
| `verify_mst_minimality_3_nodes`                               | 137.0 s           | SUCCESSFUL |
| `verify_dense_simd_dispatch_selection_respects_support_masks` | 0.65 s            | SUCCESSFUL |
| `verify_dense_simd_tail_padding_lane_bounds`                  | 3.98 s            | SUCCESSFUL |

Baseline `make kani` wall time: 1001.5 s (16 min 42 s), exit code 0, all six
harnesses successful. The wall time is dominated by Kani/cargo compilation of
`chutoro-providers-dense`, not by solving.

### `EP-M1`/`EP-M2` measured probes

| Probe                        | Configuration                                  | Outcome                                                              |
| ---------------------------- | ---------------------------------------------- | -------------------------------------------------------------------- |
| Healing harness, `unwind(4)` | kissat, concrete 2 nodes                       | FAILED on `memcmp.unwind.0` only, 5.7 s                              |
| Healing harness, `unwind(9)` | kissat, concrete 2 nodes                       | SUCCESSFUL, 9.57 s, 0/1240 failed, 2/2 covers, 0 aborts              |
| Skip reverse directed link   | mutation, `unwind(9)`                          | FAILED: `healing must link the isolated node back to the entry node` |
| Pre-seed `0 <-> 1`           | mutation, `unwind(9)`                          | FAILED: `healing precondition: both base lists must start empty`     |
| Spare-neighbour widening     | kissat, 3 nodes, 1 symbolic bool, `unwind(17)` | Stopped after 600 s+, more than 6,400 aborted paths                  |

The point of this table is the order of magnitude: a two-node HNSW harness with
symbolic edge seeding already costs about 150 s on this host. A concrete
healing harness that costs materially more than that is not a fast-tier
candidate.

## Interfaces and dependencies

No new dependency. The only new interface is the `cfg(kani)`-only bridge
`chutoro_core::hnsw::insert::ensure_base_connectivity_for_kani`, taking a
`&mut Graph`, a node index, and a connection budget. It is private to the
crate, owned by the insert module and consumed only by
`kani_proofs::connectivity`. Existing helpers reused:
`HnswParams::new_for_kani`, `Graph::with_capacity`,
`Graph::insert_first_for_kani`, `Graph::attach_node_for_kani`,
`ConnectivityHealer::new`. The bridge exists so the harness calls the
production routine instead of a copy; no parallel healing implementation is
permitted.

## Revision note

- 2026-11-21: Initial draft and reconnaissance recorded.
- 2026-11-21: `EP-M0` through `EP-M3` complete. The concrete two-node healing
  harness verifies in 9.57 s with zero aborted paths and both covers satisfied;
  the three-node symbolic widening was measured past the budget and removed;
  both negative controls fail for the intended reason. The harness is in the
  fast-tier `kani:` recipe and the documentation is updated. Remaining work is
  the final gate run, the commit, and the draft pull request.
- 2026-11-21: A first `make lint` run was stopped by its own background task
  limit while Whitaker was still compiling behind the host's compile-admission
  queue; rustdoc and Clippy had both already finished cleanly. The gate was
  restarted with a two-hour budget. This is a harness limit, not a finding.
