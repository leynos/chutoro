# Phase 1: HNSW Kani Eviction/Deferred-Scrub Scenario

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

## Purpose / Big Picture

Add a dedicated Kani harness that formally verifies the HNSW eviction and
deferred-scrub reconciliation logic. The harness pre-fills a target neighbour
list to `max_connections` capacity, forces `ensure_reverse_edge` to evict the
furthest neighbour, and asserts that bidirectional reciprocity is maintained
after `apply_deferred_scrubs` executes. Companion unit tests with broad
parameterized coverage using `rstest` validate both happy and unhappy paths. On
completion, the relevant roadmap entry in `docs/roadmap.md` is marked as done.

## Progress

- [x] (2026-01-06) Draft ExecPlan with required sections and commands.
- [x] (2026-01-06) Add Kani harness for eviction/deferred-scrub scenario.
- [x] (2026-01-06) Add unit tests with `rstest` covering eviction edge cases.
- [x] (2026-01-06) Update design documentation with decisions taken.
- [x] (2026-01-06) Update `docs/roadmap.md` to mark the entry as done.
- [x] (2026-01-06) Run formatting, linting, and tests with logging.

## Surprises & Discoveries

- Initial `multiple_evictions_in_batch_update` test failed because both updates
  used node 0 as origin, but with `max_connections = 1` at level 1 (capacity 1),
  node 0 could only have one neighbour. The second update evicted the first.
  Fixed by using different origin nodes (node 0 and node 5) so each update
  succeeds independently.

## Decision Log

- Decision: Use 4 nodes (IDs 0, 1, 2, 3) rather than 3 to allow eviction without
  affecting the new node being inserted. Rationale: With 3 nodes and one being
  the "new node", only 2 existing nodes remain, limiting eviction scenarios.
  Date/Author: 2026-01-06 (Codex)
- Decision: Use level 1 with `max_connections = 1` (capacity = 1) to trigger
  eviction with minimal nodes. Rationale: Level 0 doubles capacity
  (`2 * max_connections`), preventing eviction with few nodes. Date/Author:
  2026-01-06 (Codex)
- Decision: Use `#[kani::unwind(10)]` matching the existing commit-path harness.
  Rationale: Consistency with existing harness bounds; sufficient for 4-node
  graph traversal. Date/Author: 2026-01-06 (Codex)

## Outcomes & Retrospective

- Added Kani harness `verify_eviction_deferred_scrub_reciprocity` with 4 nodes
  and 2 levels that exercises eviction and deferred scrub logic.
- Added 5 parameterized unit tests covering eviction edge cases:
  `eviction_scrubs_orphaned_forward_edge`, `eviction_skips_scrub_if_reciprocity_restored`,
  `multiple_evictions_in_batch_update`, `eviction_at_base_layer_triggers_healing`,
  and `eviction_respects_furthest_first_ordering`.
- All 367 tests pass including the new eviction tests.
- Quality gates (`make check-fmt`, `make lint`, `make test`) all succeed.
- Roadmap entry marked as done.

## Context and Orientation

### Key Files

| File                                             | Purpose                                                                 |
| ------------------------------------------------ | ----------------------------------------------------------------------- |
| `chutoro-core/src/hnsw/kani_proofs.rs`           | Kani harnesses for HNSW invariants                                      |
| `chutoro-core/src/hnsw/insert/reconciliation.rs` | `EdgeReconciler` with `ensure_reverse_edge` and `apply_deferred_scrubs` |
| `chutoro-core/src/hnsw/insert/commit.rs`         | `CommitApplicator::apply_neighbour_updates`                             |
| `chutoro-core/src/hnsw/insert/mod.rs`            | Kani-only helpers (`apply_commit_updates_for_kani`, etc.)               |
| `chutoro-core/src/hnsw/insert/commit/tests.rs`   | Existing commit-path unit tests                                         |
| `chutoro-core/src/hnsw/insert/types.rs`          | `DeferredScrub`, `UpdateContext`, `FinalisedUpdate` types               |
| `chutoro-core/src/hnsw/invariants/mod.rs`        | `is_bidirectional` invariant checker                                    |

### Existing Harnesses

The current Kani harnesses in `kani_proofs.rs` include:

1. **Smoke test** (`verify_bidirectional_links_smoke_2_nodes_1_layer`):
   Deterministic, minimal toolchain validation.
2. **2-node reconciliation**
   (`verify_bidirectional_links_reconciliation_2_nodes_1_layer`):
   Nondeterministic edge addition with reverse-edge enforcement.
3. **3-node reconciliation**
   (`verify_bidirectional_links_reconciliation_3_nodes_1_layer`): Broader
   nondeterministic coverage.
4. **Commit-path harness** (`verify_bidirectional_links_commit_path_3_nodes`):
   Exercises production `CommitApplicator::apply_neighbour_updates` with
   deferred scrubs.

### Eviction Logic (reconciliation.rs:99-107)

When `ensure_reverse_edge` adds a reciprocal edge to a node at capacity:

1. Evicts the front entry (furthest, due to furthest-first ordering from
   trimming).
2. Pushes the new origin node to the list.
3. Creates a `DeferredScrub { origin: evicted, target, level }` request.

### Deferred Scrub Logic (reconciliation.rs:140-165)

During `apply_deferred_scrubs`:

1. For each scrub, check if `target` now links back to `origin` (a later update
   restored reciprocity) — if so, skip.
2. Otherwise, remove the orphaned forward edge from `origin → target`.
3. If removal isolates the node at the base layer, trigger connectivity
   healing.

### Roadmap Entry (docs/roadmap.md:108-110)

```markdown
- [ ] Add an eviction/deferred-scrub scenario: pre-fill a target neighbour
  list to `max_connections`, force `ensure_reverse_edge` to evict, and assert
  reciprocity after `apply_deferred_scrubs`.
```

## Plan of Work

### Step 1: Add Kani Harness for Eviction/Deferred-Scrub

Create `verify_eviction_deferred_scrub_reciprocity` in `kani_proofs.rs`:

- **Graph configuration**: 4 nodes (IDs 0, 1, 2, 3), 2 levels, `max_connections
  = 1` so level-1 capacity is 1 (triggers eviction with 2+ edges).
- **Initial state**: Seed node 1's level-1 neighbour list with node 2
  (bidirectional). This fills node 1 to capacity.
- **Update**: Node 0 adds node 1 as a neighbour at level 1. When
  `ensure_reverse_edge(ctx=0, target=1)` runs, node 1 is at capacity, so node 2
  is evicted and a `DeferredScrub { origin: 2, target: 1, level: 1 }` is
  created.
- **Post-commit**: `apply_deferred_scrubs` removes `2 → 1` (since `1` no longer
  links to `2`).
- **Assertions**:
  - `is_bidirectional(&graph)` passes.
  - Node 1 links to node 0 (not node 2).
  - Node 2 does not link to node 1 (forward edge scrubbed).

**Bounds**: `#[kani::unwind(10)]` to match the existing commit-path harness.

### Step 2: Add Parameterized Unit Tests

Extend `chutoro-core/src/hnsw/insert/commit/tests.rs` with `rstest` cases:

| Test Name                                      | Scenario                                 |
| ---------------------------------------------- | ---------------------------------------- |
| `eviction_scrubs_orphaned_forward_edge`        | Single eviction, forward edge removed    |
| `eviction_skips_scrub_if_reciprocity_restored` | Later update re-adds the reciprocal edge |
| `multiple_evictions_in_batch_update`           | Two updates each trigger eviction        |
| `eviction_at_base_layer_triggers_healing`      | Eviction isolates a node at level 0      |
| `eviction_respects_furthest_first_ordering`    | Evicts front entry (furthest)            |

Use fixtures from the existing test module (`params_two_connections`,
`insert_node`, `build_update`) and add any new helpers as needed.

### Step 3: Update Design Documentation

Record decisions in the `Decision Log` section of this document. If
architectural choices affect `docs/chutoro-design.md` or
`docs/adr-002-adoption-of-kani-formal-verification.md`, update those as well.

### Step 4: Update Roadmap

Mark the entry in `docs/roadmap.md` as done:

```markdown
- [x] Add an eviction/deferred-scrub scenario: pre-fill a target neighbour
  list to `max_connections`, force `ensure_reverse_edge` to evict, and assert
  reciprocity after `apply_deferred_scrubs`.
```

### Step 5: Run Quality Gates

```bash
set -o pipefail
make check-fmt 2>&1 | tee /tmp/make-check-fmt.log

set -o pipefail
make lint 2>&1 | tee /tmp/make-lint.log

set -o pipefail
make test 2>&1 | tee /tmp/make-test.log
```

If documentation changes are made:

```bash
set -o pipefail
make fmt 2>&1 | tee /tmp/make-fmt.log

set -o pipefail
make markdownlint 2>&1 | tee /tmp/make-markdownlint.log
```

## Concrete Steps

1. **Read** `kani_proofs.rs` to confirm harness patterns (unwind depth, helper
   imports, assertion style).

2. **Add harness** `verify_eviction_deferred_scrub_reciprocity`:

   ```rust
   #[kani::proof]
   #[kani::unwind(10)]
   fn verify_eviction_deferred_scrub_reciprocity() {
       // 4-node, 2-level graph with max_connections = 1
       let params = HnswParams::new(1, 2).expect("params must be valid");
       let max_connections = params.max_connections();
       let mut graph = Graph::with_capacity(params, 4);

       // Insert nodes 0..3 at level 1
       graph.insert_first(NodeContext { node: 0, level: 1, sequence: 0 })
           .expect("insert node 0");
       for (id, seq) in [(1, 1), (2, 2), (3, 3)] {
           graph.attach_node(NodeContext { node: id, level: 1, sequence: seq })
               .expect("attach node");
       }

       // Seed node 1 at capacity with node 2 (bidirectional)
       add_edge_if_missing(&mut graph, 1, 2, 1);
       add_edge_if_missing(&mut graph, 2, 1, 1);

       // Update: node 0 links to node 1 at level 1
       let update_ctx = EdgeContext { level: 1, max_connections };
       let staged = StagedUpdate {
           node: 0,
           ctx: update_ctx,
           candidates: vec![1],
       };
       let updates: Vec<FinalisedUpdate> = vec![(staged, vec![1])];
       let new_node = NewNodeContext { id: 3, level: 1 };

       apply_commit_updates_for_kani(&mut graph, max_connections, new_node, updates)
           .expect("commit-path updates must succeed");

       // Assert bidirectional invariant
       kani::assert(
           is_bidirectional(&graph),
           "bidirectional invariant violated after eviction and deferred scrub",
       );

       // Assert node 1 links to node 0, not node 2
       let node1_has_node0 = graph.node(1)
           .map(|n| n.neighbours(1).contains(&0))
           .unwrap_or(false);
       kani::assert(node1_has_node0, "node 1 should link to node 0 after eviction");

       // Assert node 2's forward edge to node 1 was scrubbed
       let node2_has_node1 = graph.node(2)
           .map(|n| n.neighbours(1).contains(&1))
           .unwrap_or(false);
       kani::assert(
           !node2_has_node1,
           "deferred scrub should remove node 2's forward edge to node 1",
       );
   }
   ```

3. **Add unit tests** in `commit/tests.rs`:

   - `eviction_scrubs_orphaned_forward_edge`: Verify the happy path where
     eviction triggers a scrub.
   - `eviction_skips_scrub_if_reciprocity_restored`: Construct a batch where a
     later update re-adds the reciprocal edge; verify no scrub occurs.
   - `multiple_evictions_in_batch_update`: Two updates each evict a different
     node; verify both forward edges are scrubbed.
   - `eviction_at_base_layer_triggers_healing`: Eviction at level 0 isolates a
     node; verify connectivity healing restores a link.
   - `eviction_respects_furthest_first_ordering`: Pre-fill with known ordering;
     verify front entry is evicted.

4. **Update documentation** (this file and roadmap).

5. **Run quality gates** and capture logs.

## Validation and Acceptance

- [ ] Kani harness `verify_eviction_deferred_scrub_reciprocity` passes with
  `cargo kani -p chutoro-core --harness verify_eviction_deferred_scrub_reciprocity`.
- [ ] All new `rstest` unit tests pass with `make test`.
- [ ] `make check-fmt`, `make lint`, and `make test` succeed (logs in `/tmp/`).
- [ ] Roadmap entry marked as done.
- [ ] Decision log updated with any choices made during implementation.

## Idempotence and Recovery

All steps are safe to rerun. If a test or lint step fails, fix the reported
issue and rerun the specific command with the same `set -o pipefail | tee`
pattern. If the Kani harness becomes too slow, reduce bounds or add
`kani::assume` constraints and record the decision in the Decision Log.

## Artifacts and Notes

Keep log files in `/tmp/` until the change is accepted. When citing results,
include the harness name and the command used.

## Interfaces and Dependencies

- **Input**: `CommitApplicator::apply_neighbour_updates` via the Kani helper
  `apply_commit_updates_for_kani`.
- **Invariant checker**: `crate::hnsw::invariants::is_bidirectional`.
- **Test helpers**: `add_edge_if_missing`, `assert_no_edge` from
  `insert/test_helpers.rs`.
- **Fixtures**: `params_two_connections`, `insert_node`, `build_update` from
  `commit/tests.rs`.

## References

- `docs/roadmap.md` (Phase 1, Kani harness entry)
- `docs/property-testing-design.md` (invariant definitions)
- `docs/rust-testing-with-rstest-fixtures.md` (test patterns)
- `docs/complexity-antipatterns-and-refactoring-strategies.md` (code style)
