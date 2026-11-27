# FV07: Require Kani in the merge ruleset and make proof discovery reliable

This ExecPlan is a living document. The sections `Constraints`, `Tolerances`,
`Risks`, `Progress`, `Surprises & discoveries`, `Decision log`, and
`Outcomes & retrospective` must be kept up to date as work proceeds.

Status: COMPLETE (repository side; the live ruleset edit remains a post-merge
administrator action)

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
      sources, parameterized the checker's rejection cases, and corrected the
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
      their counts; the duplicated fast/full tests are one parameterized table;
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
- [x] Fifth review round: CodeScene. The PR gate failed on nine changed files
      with two below 9.00, three Bumpy Road or Deep Nested Complexity findings,
      and ten advisory findings. Each was fixed in the code rather than
      suppressed, and `cs check` now reports 10.00 for every changed file:
      the reader's `_parse` moved its lexing into `_Lexed`/`_lex` and its item
      construction into `_module_item`/`_module_end`; the harness reader
      reports a revisit from `_note_revisit` and follows one declaration from
      `_follow_item`; its manifest walk is two predicates; the required-context
      contract reads jobs through `_document_jobs` and collisions through
      `_name_collisions`; the checker's reject/accept helpers share one
      `expect_outcome` over a `Stream`; the manifest validation is
      `_checked_document`, `_harness_list`, `_check_keys`, `_check_values`,
      `_check_tier` and `_check_qualified`; the output parser is a `Run` state
      machine over `start`/`complete`/`marker`/`finish`; `assignment_of` tests
      its name through `is_variable_name`; the recipe reading names short
      options through `_is_short_flag`; and the routing test reads a tier's
      fixture and scope from one `TIERS` table, which drops it below the
      argument limit. Two further refusals could not be fixed that way: the
      string-argument rule is a per-file ratio over twelve or more
      functions — measured against the CLI by generating files, not assumed —
      so `parser.rs` and `output.rs` split by responsibility into
      `words.rs` (what one argument spells) and `line.rs` (what one output line
      means), which leaves both under the threshold and each module with one
      job.
- [x] Sixth review round, all eight findings actioned. Three were real
      reader defects: adjacent attributes on one line (`#[cfg(kani)]
      #[kani::proof]`) were read as one attribute, so the proof was missed;
      a module's attribute lookup walked back past the previous item's
      semicolon, so `pub mod second;` adopted the earlier declaration's
      `#[path]`; and a repeated proof-free module that reaches a proof-bearing
      child dropped that child's second identity without a report. Also: shell
      comments are now stripped before `selections()` and `violations()` read
      arguments; the checker's build names the host triple explicitly and runs
      from that triple's directory, so a `build.target` in a user
      configuration cannot leave the recipes running a stale binary; the
      manifest value is quoted; the design document's declaration count is 18;
      and the CLI test's fixture moved to `tests/kani_suite_support/mod.rs`,
      which puts the test file under 400 lines.
- [x] Seventh review round, all eight findings actioned. Four were reader
      defects: an attribute reader that stopped at a line beginning with
      `mod`, losing `mod proofs { #[kani::proof]`; a `#[path]` in a raw string
      that parsed as no attribute at all, silently falling back to the default
      location; an unsupported `#[path]` shape that did the same, now reported
      as `path-attribute`; and a crate-root inventory that read no binary
      target, so a binary-only package's proofs vanished. Manifest validation
      now accepts a root-level harness name, which is the identity the reader
      reports for a proof at a crate's root. The checker build publishes by
      rename, so a concurrent `make` cannot hit `ETXTBSY` or observe a partial
      file; the CLI fixture creates its directory exclusively, so an
      interrupted run's leftovers are never adopted; and `_brace_depths` and
      `_matching_brace` moved into `rust_source_reading`, the suite's one Rust
      lexer. The adversarial reader cases moved to
      `kani_reader_fixtures_test.py`, which keeps both modules under the
      file-length limit.
- [x] Eighth review round, all six findings actioned. Three were reader
      defects in the module path: `#[cfg_attr(kani, path = "...")]` was read
      as no path at all, so discovery fell back to the default location and
      could miss a proof outside `src`, and a doc string naming
      `#[path = "x.rs"]` was read as a real path attribute. Both now inspect
      each outer attribute's own path: a conditional path is reported, and a
      mention inside another attribute is ignored. The orphan sweep now covers
      a custom `[lib] path` directory, not only `src`. The delegated command
      accepts Cargo's global options before the subcommand (`cargo --config
      net.offline=true kani`), the CLI fixture creates its directory before
      constructing the cleanup owner so an existing directory is never
      removed, and the required-context contract's description separates the
      policy it checks from the live enforcement the guide records.
- [x] Ninth review round, all five findings actioned, every one latent
      rather than live. The checker now refuses `--bin`/`--bin=...`/`--bins`
      as well as requiring `--lib`: Kani reports a binary target's harnesses
      under the same names, so a binary could satisfy a verdict after the
      library's proof disappeared. It also reads the whole value list of a
      `-p`/`--package` option, so `-p pkg other` is rejected instead of
      silently checking only `pkg`. Cargo's own global options move into
      `words.rs` with `-p` handling, which keeps `runner.rs` under 400 lines.
      The tier-reading patterns accept a column-0 comment among recipe lines,
      as make does, so nothing after one drops out of the checked recipe; the
      prover-job contract refuses a job-level `if` or `continue-on-error` on
      the nightly job as well as its steps; and the required-context contract
      checks that each owning trigger's branch filter reaches `main`.
- [x] Tenth review round, all 21 findings actioned. The substantive ones
      close further ways a run could pass as the wrong proof: a compact
      `-pkg` was unrecorded (so a second package looked like a valid command),
      a proof behind `cfg(test)` was certified though the gate never compiles
      it, `autobins = false` was ignored, a macro body delimited by
      parentheses or brackets was not seen, `mod` split over lines was not
      read, and an inline module's unreadable `#[path]` was skipped rather
      than reported. Attribute brackets are now counted in the
      literal-blanked text, so a doc string's bracket cannot move an
      attribute's start. The recipe reader drops the continuation's
      recipe-prefix tab as make does and recognizes `||   true` as masking;
      the orphan sweep covers every crate root's directories; the manifest
      walk cannot loop on a directory symlink. The contracts reject a
      manifest version the Rust loader refuses, evaluate branch filters as
      globs with ordered exclusions, require the activity types that matter,
      allow `always()` only on the closing disk reading, and compare selectors
      per invocation. The reader fixtures split into general and path
      modules over a shared support module, and the Cargo-option reader moved
      to `cargo_options.rs`, which keeps every changed file inside its bounds.
- [x] Eleventh review round, all 16 findings actioned. The substantive one
      closes the last route to a false pass: a proof behind a *module's*
      `cfg(all(kani, test))` guard now inherits that guard, and a proof in a
      binary target is reported as `binary-target` because `cargo kani --lib`
      never compiles it, so neither can satisfy a manifest entry. `autolib =
      false` stops `src/lib.rs` being read as a second crate root, the orphan
      sweep prunes build directories and covers every root, and the discovery
      root is normalized so `discover(Path("."))` reports rather than raising.
      The recipe reader matches `kani` as a whole subcommand, rejects any
      unquoted shell control operator (`|| true ;` no longer masks), and reads
      quoted option values as the shell would. Branch filters follow GitHub's
      syntax (`?`/`+` quantify the preceding character, `[]` classes), the
      manifest rejects a version that is not the integer the Rust loader
      reads, and the distance-proof count is scoped to its package. The
      fixtures gained a `write_package` factory, parameterized spellings, a
      carried-guard case, an autolib case and a pruning case.
- [x] Twelfth review round, all 14 findings actioned except two recorded as
      declined with evidence. Actioned: `cfg` predicates are now *evaluated*
      under the gate's own configuration (`test` off, `kani` on), so
      `not(test)` and `any(kani, test)` keep their proofs while
      `all(not(feature), test)` is reported; a file's inner `#![cfg(...)]` and
      an enclosing inline module's guard both reach the proofs below them;
      `cargo -vv` is accepted, and `--package=pkg other` now reads the same
      list as its separated spelling; the comment stripper is quote-aware so a
      `#` inside a value hides no option; selectors are read from shell words,
      so a quoted selector counts; a `kani:` header carrying a command after
      a semicolon is refused; the `kani` job may carry neither `if` nor
      `continue-on-error`; the source sweep ends on directory aliases; and the
      target-discovery split continues into `cargo_packages_reading.py`
      (manifest walk and package assembly) and `cargo_targets_reading.py`
      (library and binary roots). Declined: converting the CLI fixture to
      `cap_std`, because `dylint.toml` records a deliberate
      `no_std_fs_operations` exclusion for the gate-binary test crates; and
      preserving a single-quoted backslash-newline. That second reason was
      wrong, and the fourteenth round corrected it: GNU make passes the pair
      through in recipe text, and the shell decides, so `/bin/sh` keeps it
      inside single quotes. The joiner now models that.
- [x] Thirteenth review round, all three findings actioned: the checker build
      pins `--target-dir` as well as `--target`, so a `build.target-dir` in the
      caller's `CARGO` override cannot leave the copy reading a directory the
      build never wrote; `-Zvalue` and `-Cvalue` are read as one word; and a
      name-only `[[bin]]` resolves `src/bin/<name>.rs` before `autobins` is
      considered, since the declaration is the caller's while `autobins` only
      disables the targets nobody declared.
- [x] Fourteenth review round, all six findings actioned, including the
      correction of a declined reason. The Rust loader now accepts a
      crate-root harness name, matching the Python contract and the reader;
      a path-bearing inline block at the top of a stem-directory file
      resolves from the file's directory rather than nesting under the stem
      (measured against rustc 1.89: `#[path = "tf"] mod thread { mod inner; }`
      in `src/outer.rs` reads `src/tf/inner.rs`, while the same block nested
      in `mod a` reads `src/outer/a/tf/inner.rs`); a name-only `[[bin]]`
      matching the package resolves to `src/main.rs`; the fast-tier selector
      check validates identifier segments rather than requiring `::`; and the
      control-operator scan treats an unquoted backslash as an escape, so an
      escaped quote cannot hide a `;`. The continuation joiner now replaces
      an unquoted backslash-newline with a word boundary and preserves the
      pair inside single quotes, which is what make and the shell do.
- [x] Fifteenth review round, all seven findings actioned. The continuation
      joiner now models make's tab-sensitive rule — a backslash-newline plus
      the next line's recipe-prefix tab joins with nothing, while a
      continuation without a tab keeps its pair for the shell — which corrects
      the previous round's spacing. Selectors are read from shell tokens, so a
      quoted `--target-dir` cannot invent a `--harness`; the operator scan
      skips escaped characters inside double quotes; a literal `include!` is
      reported; `mod p { #![cfg(test)] }` gates its contents; a package-named
      `[[bin]]` accepts whichever conventional location exists; and an
      automatic binary an explicit `[[bin]]` claims is not read twice. The
      guard fixtures moved to `kani_guard_fixtures_test.py`, which keeps both
      modules cohesive.
- [x] Sixteenth review round, all six findings actioned. Make is measured to
      keep a recipe's backslash-newline and drop the continuation's tab, so
      the joiner now leaves the pair for the shell and removes it only outside
      single quotes; that probe also settled the previous round's spacing
      question. `include!` is detected on the literal-blanked text, so a
      mention in a string is not one, and every invocation is reported once,
      computed argument or not. An escaped space keeps a following `#` inside
      its word; `mod r#type` names the module `type`; and `src/main.rs` takes
      the package's declared name, not its directory's, when automatic targets
      are matched against explicit ones.
- [x] Push the branch and create the pull request: #474. Sixteen further
      review rounds followed on that candidate; their findings and fixes are
      recorded above and in the revision note. The live `main-required-checks`
      ruleset edit is drafted in the delivery notes and remains a post-merge
      administrator action, so this plan's repository side is complete at the
      published head.
- [x] Pre-readiness assessments at `f4d10de`. Both applicable conversations
      (ExecPlan completeness and the original issue) returned "not complete"
      with concrete gaps, each verified against the source and repaired. The
      parser now matches Kani's verdict words exactly rather than by prefix and
      parses the summary's totals grammar literally, with rejection tests for
      both; the CLI fixture records the verifier version it was captured from
      in `tests/fixtures/kani_suite_observed_version.txt` and fails when the
      pin moves without it; `nightly_kani_test.py` guards the freshness
      behaviour that had no contract; the `kani-pr.yml` header describes
      enforcement as pending rather than active; and every Python file this
      branch adds or grows is now under the 400-line limit. The three manifest
      mutation controls were run and recorded: removing an entry, adding an
      undeclared proof, and duplicating an entry each fail
      `kani_manifest_test.py` with the expected diagnostic, and each mutation
      was restored afterwards. Re-validated: `make check-fmt`, `make lint`,
      `make test` (1677 cases), `make test-workflow-contracts` (1393),
      `make kani` (seven fast harnesses).
- [x] Review validation of the assessment repairs: the strict parser was
      exercised against Kani 0.67.0's real output, which is the only evidence
      that the exact-match grammar still accepts what the verifier prints.
- [x] Second assessment round, on the file-size gap. The root `Makefile` had
      grown to 434 lines with this branch's Kani wiring, so all of it — the
      manifest and checker variables, `kani-suite-build`, and both tiers —
      moved to `tools/kani/kani.mk`, which the `Makefile` includes. `help`
      already reads `$(MAKEFILE_LIST)`, so the fragment's targets remain
      listed, and the fixture trees that copy the `Makefile` now copy the
      fragment with it. `tests/workflow_contracts/trybuild_override_test.py`,
      already 583 lines on `main`, was split by responsibility so that no
      400-line exception is needed: its discovery-sweep cases moved to
      `rust_test_discovery_test.py`.

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
the real Kani run covers the manifest's seven fast harnesses and two
package-wide full sweeps. The earlier six-harness figures in this plan are
pre-rebase evidence, taken before the trunk added the connectivity healer; the
current counts are seven fast and eighteen in the full sweep, which is the
manifest's own total.

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
own failure the pipeline's; `make kani` ends with a checker line naming the
manifest's seven fast harnesses; the workflow-contract suite reports its new
tests passing.

## Outcomes & retrospective

The repository side of FV07 is implemented and validated:

- the proof surface is classified once in `tools/kani/harnesses.toml` and
  bound in both directions to the Rust declarations and the Makefile;
- every Kani invocation is checked against that inventory, and both real tiers
  pass: after the rebase, seven fast harnesses and 18 in the full sweep;
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

Revised 2026-11-25: recorded the rebase onto a trunk that added a seventh fast
harness, and the fifth review round's CodeScene refusals and their fixes.

Revised 2026-11-26: recorded the sixth to ninth review rounds, whose findings
were the shapes the reader and the checker could not see: adjacent and
non-line-anchored attributes, raw and conditional `#[path]` values, binary
targets, repeated proof-free parents, multi-valued `-p`, and Cargo's global
options. The checker's build now names its target and publishes by rename, and
the artefacts' documentation follows.

Revised 2026-11-27: recorded the tenth to sixteenth review rounds (module `cfg`
guards, binary-target refusal, `cfg` predicate evaluation, `/bin/sh` quoting,
rustc's `#[path]` directory rule, Cargo's target and binary inference, and
make's continuation rule — each measured rather than assumed), marked the
repository side complete at published head `7aba491`, and corrected the harness
counts this plan had carried from before the rebase: seven fast harnesses and
eighteen in the full sweep, the manifest's own total.

Revised 2026-11-27 (second): recorded the pre-readiness assessments and their
repairs — exact verdict and summary grammar in the checker, a version-stamped
output fixture, a nightly-freshness contract test, the enforcement wording, and
the module splits that bring every file this branch adds or grows inside the
400-line limit. The manifest mutation controls were run and recorded, not
assumed. `tests/workflow_contracts/trybuild_override_test.py`, 583 lines on
`main` and touched here only by an import, was then split by responsibility:
its discovery-sweep cases moved to `rust_test_discovery_test.py`, leaving 377
and 244 lines, so no policy exception is needed.

Revised 2026-11-27 (third): split the Makefile's Kani wiring into
`tools/kani/kani.mk` after the second assessment round found the root file over
the 400-line limit, and recorded the one remaining over-limit file as
pre-existing on `main`.

Revised 2026-11-27 (fourth): after the third review pass, fixed `make help`,
whose column layout broke once `$(MAKEFILE_LIST)` named an include: `grep`
prefixed each match with its filename and the target slid into the description
column. `-h` and a `##`-based split restore both columns, and
`makefile_help_test.py` pins a root target and both fragment targets. Also
corrected this plan's file-size claim to say *adds or grows*, and recorded the
pre-existing over-limit file explicitly.

Revised 2026-11-27 (fifth): rather than carry a 400-line exception that only a
maintainer could approve, split `trybuild_override_test.py` by responsibility.
Its discovery-sweep cases moved to `rust_test_discovery_test.py`; both files
are now well inside the limit and the exception is withdrawn.
