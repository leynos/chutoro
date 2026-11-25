# FV07: Require Kani in the merge ruleset and make proof discovery reliable

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: IN PROGRESS

## Purpose / big picture

Kani now runs successfully on pull requests, and the full 17-harness suite
completed on source baseline `56d6909`. Historical statements that Kani never
runs, or always times out, are obsolete. The enforcement and maintenance gaps
that remain are narrower but serious:

- The live `main-required-checks` ruleset requires only `build-test` and
  `verus-proofs`; it does not require `kani`, so a failing or missing proof run
  cannot block a merge.
- The workflow-contract inventory discovers files named `kani_*.rs` and files
  under `kani_proofs/`. It misses the six proof declarations inside
  `chutoro-core/src/distance/mod.rs`, so a proof can disappear without any
  contract failing.
- The pull-request path filter omits build inputs such as the owning crate
  roots, the dense provider's `build.rs`, and Cargo configuration. A change to
  one of those can alter what the proof build compiles without starting the job.
- The nightly job decides whether to verify from commit age, not from whether
  the exact head has a successful full verification. A missed or
  infrastructure-failed run can age out and be followed by a skipped green
  workflow.

After this change a reader can observe that Kani is enforced by the
repository's own contracts: every proof declaration is classified exactly once
in a checked-in manifest, every Kani invocation is checked against that
manifest's expected result set, the pull-request job is always present, the
nightly job verifies its own checkout unconditionally, and a static contract
requires the `kani` context to be configured for every pull request. The live
ruleset edit itself is a repository-administration action requested after
merge; the repository side is complete when the workflow reports `kani` from
GitHub Actions app `15368` on every pull request.

## Conformance basis

- Issue [#396](https://github.com/leynos/chutoro/issues/396), "FV07: Require
  Kani in the merge ruleset and make proof discovery reliable", open.
- Related issue [#202](https://github.com/leynos/chutoro/issues/202), closed;
  this work preserves its result that a Kani job runs on pull requests.
- Related issue [#215](https://github.com/leynos/chutoro/issues/215), closed;
  this work distinguishes running a job from requiring it.
- [ADR-0002](../adrs/adr-0002-adoption-of-kani-formal-verification.md),
  amended 2026-11-19 for the concurrency boundary.
- [Developer's guide, "Kani CI policy"](../developers-guide.md#kani-ci-policy).
- [Historical execution plan: nightly slow CI job](phase-1-nightly-slow-ci-job.md),
  superseded by this plan.

No Terms of Reference or technical design revision governs this repository. The
issue text and the repository's documentation are the authority.

## Constraints

- Do not change the meaning of `make test`, `make lint`, `make fmt`, or the
  development loop. Kani remains an opt-in proof gate.
- Keep `tools/kani/VERSION` as the single source of truth for the verifier
  version; never restate a version literal in the workflow.
- Keep every workflow action pinned to a full commit SHA.
- The pull-request job ID stays `kani`; the required-status context must match
  that job name exactly, with no display-name override.
- Do not modify `nightly-portable-simd.yml`.
- Do not modify remote repository settings from this work. The live ruleset
  change is requested from an administrator after the workflow change merges.
- Every Rust module starts with `//!` documentation. Public items carry Rustdoc.
- No file exceeds 400 lines. All lint findings are fixed in code, not
  suppressed.
- Markdown prose is wrapped at 80 columns and code at 120 columns, in
  en-GB-oxendict spelling.
- Generated `typos.toml` is never edited by hand.
- The repository's own gate ordering applies: formatting, lint, and tests run
  sequentially, and the full commit gates are run before a commit.

## Tolerances (exception triggers)

- If the manifest or checker needs a dependency not already in the lockfile,
  prefer a workspace declaration; adding more than two direct dependencies
  requires a documented rationale in this plan.
- If Kani 0.67.0's real output cannot be parsed into the expected multiset after
  two independent parser revisions, stop and record the observed format before
  changing the approach.
- If a live `workflow_dispatch` run cannot be triggered from this worktree,
  record the blocking condition and validate the parser against a locally
  captured run instead.
- If the live ruleset API refuses the query or the identity check fails, report
  the exact error and continue with the repository-side work.
- If a required full-suite run exceeds the repository's documented 120-minute
  nightly budget on the baseline, stop and report; do not silently raise the
  timeout.
- ExecPlan files are excluded from size and line-count tolerances.

## Risks

- Risk: Kani output is human-readable and may change between versions.
  Mitigation: pin the parser to Kani 0.67.0's observed strings, bind a fixture
  version header to `tools/kani/VERSION`, and fail closed on an unrecognized
  line rather than silently accepting it.
- Risk: A manifest and a source scan can drift in opposite directions.
  Mitigation: compare package/path multisets in both directions, report added,
  removed, and duplicate identities separately, and assert every declaration
  has exactly one tier.
- Risk: Adding a required check before the workflow reports it blocks every
  pull request. Mitigation: sequence the live ruleset change after the workflow
  change merges, and verify via the effective-rules API first.
- Risk: Removing the PR path filter raises cost. Mitigation: the issue requires
  an always-present status; the job is already bounded at 30 minutes and runs
  on the paid queue. Record the trade-off in the developer's guide.
- Risk: Deleting the nightly gate removes behaviour that other contracts
  reference. Mitigation: search every reference, delete the module, binary, CLI
  test, lint exclusion, and documentation explanation together, and update the
  workflow contract in the same change.

## Progress

- [x] (2026-11-23) Record this ExecPlan and orient in the repository.
- [x] Phase 1: checked-in harness inventory and workflow-contract binding.
      `tools/kani/harnesses.toml`, `kani_harness_reading.py`,
      `kani_manifest_test.py`, and the rewritten `kani_fast_tier_test.py` are
      in place; 33 contract tests pass and the source scan finds exactly the
      17 declarations Kani's own list reports.
- [x] Phase 2: repository-owned Kani result checker and Makefile routing.
      `cargo check --all-targets` is clean, the 209-test test-support suite
      passes including the fake-command CLI tests, `make lint` passes
      (interrogate, rustdoc, Clippy, Whitaker), and a real `make kani`
      verified all six fast harnesses through the checker with exit 0.
      The checker is built once per `make` invocation rather than through
      `cargo run` on every recipe line, after the first real run showed six
      redundant builds spending the job budget on the admission queue.
- [x] Phase 3: always-present PR job and unconditional nightly verification.
      Workflows and their contracts are edited; the nightly gate module,
      binary, CLI test, export, and lint exclusion are deleted.
- [x] Phase 4 (repository side): required-context contract
      (`required_checks_test.py`) and the developer's guide updates.
- [x] Validate both tiers against Kani 0.67.0. `make kani` verified the six
      fast harnesses (four in `chutoro-core`, two in
      `chutoro-providers-dense`) and exited 0. `make kani-full` verified 15 in
      `chutoro-core` and 2 in `chutoro-providers-dense`, each run ending
      `Complete - N successfully verified harnesses, 0 failures, N total.`
      and `kani_suite: verified N harness(es) ...`.
- [x] Run the commit gates: formatting, lint (interrogate, rustdoc, Clippy,
      Whitaker), the 1587-test nextest suite, doctests, the workflow contracts,
      Markdown lint, Mermaid, and both real Kani tiers.
- [x] First CodeRabbit review, with all 13 findings actioned. Two were
      substantive corrections to this work: the guide and the design document
      claimed `kani` was already a required check when the ruleset change is
      still pending, and the #215 draft claimed Kani cannot drift when the MST
      and dense harnesses verify models rather than production paths. The rest
      tightened the contracts: attached-attribute reading, repeated-source
      identities, package-scoped duplicate detection, a non-vacuous per-command
      assertion, a display-name collision check, a bare `pull_request:` trigger,
      a required final summary, Result-returning fixtures, a Unix gate, and
      `pipefail` in the documented commands.
- [x] Second CodeRabbit review, all 16 findings actioned. Two were real
      defects: the checker's binary path was hardcoded to `target/debug`, so a
      `CARGO_TARGET_DIR` override built it where the recipes did not look for
      it, and the required-context contract did not reject a `needs:` or
      `continue-on-error` through which a required job can report skipped or
      tolerant. A third tightened the checker: a run with no `Complete - ...`
      summary is now rejected, because a record truncated after the last
      expected block would otherwise look complete. The rest accepted make's
      `@` and `+` recipe prefixes, kept blank and comment recipe lines, reused
      one continuation joiner across two modules, reported an unreadable source,
      derived the inline-module count and the tier call counts from their
      sources, parameterised the checker's rejection cases, and corrected the
      guide's checker path and the enforcement test's proof tier.
- [x] Third CodeRabbit review, all 13 findings actioned. The substantive one
      was a `#[path]` resolution bug: Rust gives a path-loaded file the
      `mod.rs` directory rule for its children, and the reader used the stem
      directory instead, so a child beside a path-loaded module would have been
      looked for somewhere Rust never uses. It is fixed and has a regression
      test. The others: the derived checker path is quoted; the delegated
      command accepts a leading `+toolchain` word; the fake verifier moved to a
      real script under `tests/fixtures/` (which also brought the test module
      under 400 lines); the event helper accepts scalar and sequence `on:`
      forms and names the file on a genuine error; the package walk prunes
      ignored directories instead of filtering after traversing them; the
      wrapper and dry-run contracts accept make's recipe prefixes and derive
      their counts; the duplicated fast/full tests are one parameterised table;
      the failure-propagation test runs both tiers; bare asserts carry failure
      messages; and the draft keeps Kani's enforcement explicitly pending.
      Whitaker then caught the runner module at 405 lines from the `+toolchain`
      addition, so its argument walk moved to `parser.rs`.
- [x] Fourth CodeRabbit review, all 10 findings actioned, and every one is a
      real gap. The reader now reads attributes by offset rather than by line,
      so `#[path = "x"] mod m;` on one line and a `#[path]` spread over three
      lines are both attached; it stops the walk at a brace, so an enclosing
      inline module's attribute cannot leak to its first child; it applies an
      inline module's own `#[path]` to its children's directory, as Rust does;
      it discovers packages whose library lives at an explicit `[lib] path`;
      and the repeated-source report now fires only for proof-bearing files, so
      a shared test-only source is not a false positive. Four regression tests
      cover the reader. The checker refuses `-j4` as well as `-j`, the recipe
      reading rejects `|| true # comment`, the fixture owns its directory
      before the first fallible write, and the Makefile clears
      `CARGO_BUILD_TARGET` for the checker build so a configured triple cannot
      leave the proofs running a path Cargo never wrote. The guide no longer
      calls Kani "equally enforced" and states that a full-tier proof outside
      the two named packages needs a `kani-full` sweep entry.
- [ ] Push the branch and create the draft pull request.

## Surprises & discoveries

- Observation: `distance/mod.rs` declares six proofs in an inline
  `#[cfg(kani)] mod kani_proofs { ... }` block. The existing inventory scans
  filenames only, so these six were invisible to every contract.
- Observation: the fast tier is six selectors across three modules
  (two HNSW bidirectional, two MST, two dense SIMD), while the full tier adds
  six distance proofs and five HNSW invariant proofs. That is 17 declarations
  in total, matching the ADR's "17 harnesses". `cargo kani list --format json`
  reports the same 15 core harnesses and the same two dense ones with the same
  qualified paths the manifest records.
- Observation: Kani 0.67.0's CLI exposes `--output-format regular|terse|old`,
  `--output-into-files`, `--only-codegen`, `--no-codegen`, `--log-file`, and
  `--quiet`. Any of these can suppress or redirect the harness blocks the
  checker parses, so the checker rejects them before execution.
- Observation: Kani's `Manual Harness Summary` prints
  `Complete - {succeeded} successfully verified harnesses, {failed} failures,
  {total} total.`
  and a `Verification failed for - <name>` line per failure, and
  `determine_targets` fails a zero-match `--harness` filter before codegen. A
  filter that matches a *subset* is not an error; that is the case the
  checker's expected multiset closes.
- Observation: a `#[path = "..."]` value is a string literal, so the fully
  blanked source the readers use erases the value they need. The reader keeps a
  comment-blanked copy for attribute values. A resolvable path attribute still
  yields the module's name from the declaration, which is what Rust does, so
  the repository's ordinary test-only `#[path]` modules are readable rather
  than reported.
- Observation: `thiserror` treats a field named `source` as an error source, so
  a `String` field of that name does not compile. The diagnostic fields are
  named `detail`.

## Decision log

- Decision: Represent the proof inventory as a checked-in TOML manifest with
  `package`, `pretty_name`, and `tier` per harness. Rationale: a single
  classification surface can be consumed by Python workflow contracts and by
  the Rust checker, and it keeps tier decisions visible in review. Date/Author:
  2026-11-23 (agent).
- Decision: Derive source declarations by scanning Rust files rather than
  consuming `cargo kani list`. Rationale: the contract must fail when a proof
  declaration disappears, and a verifier inventory generated from the same
  broken build would not catch a source-level loss. `cargo kani list` is used
  as an independent cross-check during this work rather than as an input.
  Date/Author: 2026-11-23 (agent).
- Decision: Route every Makefile Kani invocation through a repository-owned
  checker that parses the regular output and requires the exact successful
  multiset from the manifest. Rationale: Kani exits successfully when it
  verifies fewer harnesses than requested or when a selector matches nothing;
  the checker makes the empty and reduced cases fail. Date/Author: 2026-11-23
  (agent).
- Decision: Remove the pull-request path filter entirely instead of widening
  it. Rationale: the issue's first closure item requires an always-present
  status. Change detection inside the workflow would need a second job or a
  conditional that can still skip the required context; an always-running
  30-minute bounded job is simpler and cannot produce a missing status.
  Date/Author: 2026-11-23 (agent).
- Decision: Replace the nightly age gate with an unconditional run on
  `github.sha`, recording `git rev-parse HEAD`. Rationale: a skipped, failed,
  absent, or cancelled run must not count as proof success, and only a run of
  the exact head can establish that the head was verified. Date/Author:
  2026-11-23 (agent).
- Decision: Make the fast-tier Makefile selectors fully qualified rather than
  bare function names. Rationale: `--exact` compares the whole harness path, so
  the qualified form is what the manifest, the checker, and Kani's own
  `pretty_name` share; a bare name would silently depend on substring matching.
  Date/Author: 2026-11-23 (agent).
- Decision: Let a `#[path]`-mediated module contribute its declaration's name
  rather than forcing it to be reported. Rationale: Rust takes the module path
  from the declaration, so the identity stays derivable and the repository's
  test-only `#[path]` modules do not become false positives. An unresolvable
  path and a proof-bearing source no module reaches are still reported.
  Date/Author: 2026-11-23 (agent).

## Context and orientation

The repository is a Rust workspace. Kani proof harnesses live beside the code
they verify:

- `chutoro-core/src/distance/mod.rs` holds six inline distance proofs under
  `#[cfg(kani)] mod kani_proofs`.
- `chutoro-core/src/hnsw/kani_proofs/` holds seven HNSW proofs: two
  bidirectional fast proofs in `bidirectional.rs`, and five invariant proofs in
  `invariants.rs` (two no-self-loop, two neighbour-uniqueness, one
  entry-point-validity).
- `chutoro-core/src/mst/kani_harness.rs` holds two fast MST proofs.
- `chutoro-providers/dense/src/simd/kani_proofs.rs` holds two fast dense SIMD
  proofs.

The Makefile defines `kani` (fast tier) and `kani-full` (package-wide sweep)
targets, deriving `KANI_VERSION` from `tools/kani/VERSION` and passing
`KANI_ENV` (`LD_LIBRARY_PATH`) to `cargo kani`.

Workflow contracts live in `tests/workflow_contracts/` and run under
`make test-workflow-contracts`. They parse workflow YAML and source files
directly. `rust_source_reading.py` already provides a lexer that distinguishes
code from comments, string literals, raw strings, byte strings, and character
literals; `rust_declaration_reading.py` already counts brace depth and locates
inline module bodies. This work extends those readers rather than growing a
second scanner.

The pull-request workflow `.github/workflows/kani-pr.yml` defines the single job
`kani`, which runs `make kani` and records disk headroom around the suite. The
nightly workflow `.github/workflows/nightly-kani.yml` defines `kani-full`,
gated today by `chutoro-test-support/src/ci/nightly_gate.rs` and its
`kani_nightly_gate` binary.

## Plan of work

### Phase 1: checked-in inventory and source binding

End state: `tools/kani/harnesses.toml` lists all 17 declarations with package,
qualified `pretty_name`, and tier. `tests/workflow_contracts/` gains a Rust
declaration reader that reports file-based and nested inline module paths,
ignores comments and literals, recognizes `#[kani::proof]` and
`#[kani::proof_for_contract]`, and fails on unresolved `#[path]`,
`cfg_attr`-wrapped proofs, and macro-contained proofs. A manifest contract
compares package/path multisets both ways, reports added, removed, and
duplicate identities separately, and asserts package-set equality.
`kani_fast_tier_test.py` is rewritten to compare the exact fast selectors and
package-scoped full sweeps with the manifest, rejecting masked recipe lines
while retaining commented and echoed command fixtures.

Acceptance evidence: `make test-workflow-contracts` passes; removing a manifest
entry fails with a removed-identity report; adding a fake proof fails with an
added-identity report; duplicating an entry fails with a duplicate report; the
six distance proofs are discovered.

### Phase 2: repository-owned result checker

End state: `chutoro-test-support/src/ci/kani_suite.rs` provides pure parsing
and comparison over Kani's regular output, and a thin `kani_suite` binary
accepts `--manifest`, `--package`, `--scope fast|full`, and a command after
`--`. The checker:

- validates the manifest (unknown keys, unknown tiers, missing fields,
  duplicate identities);
- rejects forbidden flags (`--quiet`, `--output-format` other than `regular`,
  `--output-into-files`, `--no-codegen`, `--only-codegen`, `--log-file`,
  `--tests`, `-j`) and non-regular output formats before execution;
- derives the expected harness set from the manifest and the requested scope;
- runs the delegated command with any leading `NAME=VALUE` words applied only
  to the child;
- parses sequential `Checking harness <name>...` blocks and their success or
  failure outcomes;
- rejects a nonzero delegated status, failed or incomplete blocks, duplicate
  names, missing expected names, and unexpected names;
- cross-checks the `Manual Harness Summary` totals when present;
- reports every category separately and fails closed.

The Makefile routes every `cargo kani` call through `$(KANI_SUITE)`, passes
`KANI_ENV` only to the delegated process, adds `--lib`, uses exact qualified
fast selectors, and retains the unwind values. Unit and fake-command CLI tests
cover ordered calls, first/middle/last failure propagation, and dry-run routing.

Acceptance evidence: `cargo nextest run -p chutoro-test-support` passes;
`make kani` succeeds against the real Kani 0.67.0 installation; a removed
manifest entry fails before or during verification; a stubbed failing harness
propagates its failure.

### Phase 3: always-present PR job and unconditional nightly run

End state: `.github/workflows/kani-pr.yml` has no `paths` filter and keeps the
`kani` job, setup, install, proof, and disk-diagnostic steps unconditional and
blocking. `.github/workflows/nightly-kani.yml` has no gate, no `force_run`
input, and no `should_run` conditions; it checks out `${{ github.sha }}`,
records `git rev-parse HEAD` in `GITHUB_STEP_SUMMARY`, then installs Kani and
runs `make kani-full`. The nightly gate module, binary, CLI test, module
export, lint exclusion, and code-health explanation are deleted.
`kani_pr_test.py` and `prover_setup_test.py` are updated accordingly.

Acceptance evidence: `make test-workflow-contracts` passes; no reference to
`should_run`, `force_run`, `nightly_gate`, or `kani_nightly_gate` remains;
`nightly-portable-simd.yml` is unchanged.

### Phase 4: required-context contract and documentation

End state: `tests/workflow_contracts/required_checks_test.py` requires each of
`build-test`, `verus-proofs`, and `kani` to map to exactly one pull-request job
ID, with no name override, matrix, or job-level condition, and compares the
list with `docs/developers-guide.md`. The developer's guide, design document,
roadmap, and ADR-0002 gain the new policy. The historical nightly execution
plan is marked superseded without rewriting it. A draft comment for #215 and a
draft ruleset-change request are recorded with the verification steps for the
post-merge administrator action.

Acceptance evidence: `make test-workflow-contracts` passes; the developer's
guide names the three required contexts and the manifest/checker procedure; the
draft requests are reviewable as text; the live ruleset is left unchanged.

## Verification plan

The work introduces these invariants and obligations:

1. Every proof declaration in the tree appears exactly once in the manifest
   with the same package and qualified path. Verified by a Python contract over
   parsed Rust sources with adversarial fixtures for comments, literals, inline
   modules, `#[path]`, `cfg_attr`, and macro-contained proofs. A removed or
   added declaration is reported, not silently accepted. Negative controls:
   remove a manifest entry, add a fake proof, duplicate an entry.
2. The fast tier in the Makefile is exactly the manifest's fast set for each
   package, and the full tier is the package-scoped sweep. Verified by
   `kani_fast_tier_test.py`, which rejects comments and echoed text as
   selectors. Negative control: replace a selector with an echo line.
3. A Kani run is successful only when the exact expected multiset of harnesses
   completed successfully. Verified by pure unit tests over captured regular
   output, CLI tests against a fake command, and a real run against Kani
   0.67.0. Failures cover a missing name, an unexpected name, a duplicate name,
   a failed harness, an incomplete block, a delegated nonzero exit, and a
   summary-total mismatch.
4. The pull-request `kani` job always reports. Verified statically by
   `kani_pr_test.py` (no `paths`, unconditional job, unconditional steps) and
   dynamically by a workflow dispatch or pull-request run showing the `kani`
   context from app `15368`.
5. The nightly job verifies its own checkout unconditionally. Verified
   statically by `prover_setup_test.py` (checkout ref `${{ github.sha }}`, no
   gate references, blocking steps) and by the workflow's recorded SHA line.
6. The required-context list is complete and matches the documentation.
   Verified statically by `required_checks_test.py`, and post-merge by the
   effective-rules API and a failing-check experiment on a non-bypass account
   (administrator action; drafted here, not executed).

Axioms: GitHub Actions reports a job's name as its required-status context;
GitHub's effective-rules API returns the union of rulesets; app `15368` is the
GitHub Actions integration; Kani 0.67.0's regular output uses the strings the
parser pins. Each axiom is exercised by the real run or the live API query
where possible, and otherwise recorded as an external assumption.

Non-vacuity: each contract has at least one seeded failure that it rejects, and
the real Kani run covers six fast harnesses and two package-wide full sweeps.

## Validation commands

Run sequentially from the repository root, capturing output under `/tmp`.
`pipefail` is not decoration: a bare `make … | tee …` reports `tee`'s status,
so a failing gate would leave the pipeline green and the evidence would say the
opposite of what happened.

```sh
set -euo pipefail
make fmt 2>&1 | tee /tmp/fmt-issue-396.out
make check-fmt 2>&1 | tee /tmp/check-fmt-issue-396.out
make lint 2>&1 | tee /tmp/lint-issue-396.out
make test 2>&1 | tee /tmp/test-issue-396.out
make test-workflow-contracts 2>&1 | tee /tmp/workflow-contracts-issue-396.out
make markdownlint 2>&1 | tee /tmp/markdownlint-issue-396.out
make nixie 2>&1 | tee /tmp/nixie-issue-396.out
make kani 2>&1 | tee /tmp/kani-issue-396.out
```

Expected evidence: every command exits zero, with `pipefail` making a gate's
own failure the pipeline's; `make kani` ends with a checker line naming six
successful harnesses; the workflow-contract suite reports its new tests passing.

## Outcomes & retrospective

The repository side of FV07 is implemented and validated:

- the proof surface is classified once in `tools/kani/harnesses.toml` and
  bound in both directions to the Rust declarations and the Makefile;
- every Kani invocation is checked against that inventory, and both real tiers
  pass: six fast harnesses and 15 plus 2 full harnesses;
- the pull-request `kani` job reports on every change and the nightly job
  verifies its own SHA unconditionally;
- the required-context contract and the documentation agree on
  `build-test`, `verus-proofs` and `kani`.

What is deliberately left for the post-merge administrator action is the live
ruleset edit, which is drafted in
[the delivery notes](../planning/issue-396-ruleset-change-request.md) with its
preconditions and verification. The #215 comment is drafted there too; neither
has been posted.

Lessons: Kani's exit status is not a count, so a run checker has to compare the
result multiset rather than trust the process; a filename-based inventory
misses inline modules, so the source has to be read; and a checker invoked
through `cargo run` per recipe line pays its own build cost once per proof, so
it is built once per `make` invocation instead.

## Revision note

Initial plan recorded 2026-11-23 from issue #396 and the supplied coding plan.
Revised 2026-11-23 after Phase 1: recorded the collected observations that
changed two decisions (qualified Makefile selectors, and path-mediated modules
keeping their declaration's name), and noted the `thiserror` field-name
constraint.
