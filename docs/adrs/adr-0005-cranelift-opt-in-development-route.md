# Architecture Decision Record (ADR) 005: Cranelift as an opt-in development route

## Status

Accepted - 2026-11-06: Cranelift stays outside the build standard, reachable
only through the explicit `dev-build` and `dev-test` targets.

## Date

2026-11-06

## Context and problem statement

Debug builds dominate the edit-compile-test loop, and the Cranelift codegen
backend compiles unoptimized Rust substantially faster than LLVM. The
repository's build standard, adopted in `d167c04` ("Adopt the Rust build
standard (#307)"), pins stable Rust 1.99.1 and routes every Cargo gate through
`.cargo/config.toml`. Cranelift requires a nightly toolchain.

A proposal to make Cranelift the development-profile backend would have routed
the standard `build`, `test`, `typecheck`, and `lint` targets through a nightly
pin. That contradicts the build standard the repository had already adopted, and
`docs/developers-guide.md` records the standard's position that Cranelift is
not the development-profile backend.

The question this decision answers is how to keep the compile-speed benefit
available without letting a nightly toolchain reach a merge gate.

## Decision drivers

- A Cargo merge gate must never execute on a toolchain other than the pinned
  stable release, so a gate result stays reproducible and matches what
  consumers run. The verifier gates keep their own pinned toolchains, which is
  a property of those verifiers rather than a route this decision opens.
- Cargo selects one `rustflags` source rather than merging them, so any route
  that assigns `RUSTFLAGS` silently replaces the documentation denials and the
  Linux linker flag configured in `.cargo/config.toml`.
- A developer-speed route must be discoverable and one command away, or it goes
  unused.
- Adding a nightly toolchain must not add provisioning to CI, because no gate
  needs it.

## Options considered

- **Cranelift as the development-profile backend.** Rejected. It requires a
  nightly pin, so the standard targets would stop matching the build standard
  and every gate would run on a toolchain the project does not pin.
- **Cranelift reachable only through Cargo's automatic configuration paths.**
  Rejected. A fragment placed where Cargo finds it automatically activates for
  every invocation in the directory, including gates.
- **Cranelift behind explicit, named Make targets.** Accepted. Activation
  depends on an argument a caller passes, so nothing selects it by accident.
- **No accelerated route at all.** Rejected. It discards a real productivity
  gain for a hazard that a named boundary removes.

## Decision outcome

The project keeps Cranelift as an opt-in development route reached only through
`make dev-build` and `make dev-test`. Every standard Cargo gate — `build`,
`test`, `typecheck`, `lint-clippy`, `release`, formatting, coverage, and
benchmarks — stays on the pinned stable 1.99.1 and reads `.cargo/config.toml`.

Kani, Verus, and Whitaker are the exceptions. Kani and Verus are not Cargo
gates in this sense, and each runs on the toolchain its own verifier release
requires: `make install-kani` and `make install-verus` provision from the pinned
`rust-prover-tools` package, `make kani-check` reads the installed verifier's
version through it, and `make verus` runs the proofs through it. `make kani` is
the one exception to that delegation: it invokes `$(CARGO) kani` directly,
because `cargo-kani` is the verifier's own front-end rather than something the
shared package wraps. Whitaker runs under the toolchain its own installer
manages rather than the repository pin. What the decision protects is unchanged
— none of the three is reached through the development fragment, and none can
select Cranelift.

The fragment lives in `tools/dev-fast/config.toml`, outside Cargo's automatic
configuration paths, and is pinned to one dated nightly in
`tools/dev-fast/TOOLCHAIN`. The Makefile's `DEV_CARGO` is the only thing that
names it, through an explicit `--config` argument, so installing the nightly
components alone does not activate it.

`make install-dev-fast` provisions the toolchain components and, on Linux, the
checksum-verified `mold` 2.41.0 archive for x86_64 and aarch64 hosts. CI
provisions nothing for the fragment. `mold` itself is shared rather than
development-only: `.cargo/config.toml` selects it in the Linux `rustflags`
table, so the standard Linux `build`, `test`, `typecheck`, and `lint-clippy`
routes need it too, and CI installs it for those routes.

Because nextest starts a separate Cargo invocation that does not inherit the
outer configuration, `dev-test` passes the fragment before `nextest` and again
to `nextest run`. Both development routes assign `RUSTFLAGS`, appending to any
inherited value rather than replacing it, and therefore repeat the
documentation denials and the Linux linker flag that the assignment would
otherwise replace; `dev-test` also denies warnings. The host probe is evaluated
once at parse time, together with its exit status, and every later reference
reads that captured answer. The target fails when the probe reports no
operating system or exits non-zero, rather than silently selecting the
non-Linux branch.

## Architectural rationale

The decision turns a toolchain choice into an explicit boundary rather than an
implicit default. A named target states its dependency on a nightly toolchain
in its own name, so a reader of a CI workflow or a gate cannot reach it without
noticing. The contract tests in `tests/workflow_contracts/dev_fast_*` pin both
halves: the development targets select the fragment, and every standard target
stays off it.

This follows the project's hexagonal instinct that infrastructure choices
belong behind an explicit port rather than leaking into the domain. Here the
"domain" is the set of gates that decide mergeability, and the "adapter" is the
optional backend.

## Known risks and limitations

- The development route can drift from the standard route, because it compiles
  with a different backend. `dev-test` is a convenience, not evidence of what a
  merge gate will observe.
- The nightly pin is a moving target in the sense that nightly releases are
  replaced; the pin is what makes it reproducible, and it needs periodic
  refresh.
- The Linux developer route additionally depends on GCC 12.1 or later, because
  earlier releases reject `-fuse-ld=mold`.
- `dev-build` and `dev-test` are outside `build_standard_contract.rs`; a change
  to the build standard does not automatically reach them.

## Consequences

- Enabling the speed-up is one explicit command, and it cannot affect a gate.
- A result from a Cargo gate is always attributable to the pinned stable
  toolchain. The verifier gates are the standing exceptions, and this decision
  neither creates nor removes them.
- The fragment duplicates the documentation denials and the linker flag, and a
  contract test fails if that duplication is dropped.

## Addendum — 2026-11-07

Two statements in the original decision text did not match the repository, and
the pre-merge review of PR #289 caught both.

The first named `scripts/run-verus.sh` and its `VERUS_BIN` as the route
`make verus` took. No such script exists. Prover acquisition moved into the
pinned `rust-prover-tools` package, so `make install-kani`,
`make install-verus`, and `make verus` invoke `prover-tools`, and
`tests/workflow_contracts/tool_install_test.py` lists the three former scripts
among the tools whose acquisition that package absorbed.

The second counted Whitaker among the gates on the pinned stable 1.99.1.
`make lint-whitaker` assigns `RUSTFLAGS` and runs the `whitaker` wrapper under
the toolchain its own installer manages, exactly so the driver build is not
subject to the repository pin or the development standard. It is a verifier
exception beside Kani and Verus, not a Cargo gate, and the decision outcome now
says so.

A third over-statement reached the `Decision outcome` paragraph while the first
two were being fixed: it said both `make kani` and `make verus` invoke the
shared package. Only `make verus` does. `make kani` runs `$(CARGO) kani` for
each harness directly, because `cargo-kani` is the verifier's own front-end,
and the delegation table in `tool_install_test.py` lists exactly four targets:
`install-kani`, `install-verus`, `verus`, and `kani-check`. The paragraph above
now separates them.

None of the corrections changes the decision. The boundary this record protects
is that no gate selects Cranelift, and none of the three exceptions is
reachable through the development fragment.

## Addendum — 2026-11-10

Two sentences in the original text were still wider than the repository, and
the pre-merge review of #289 caught both on the third pass.

The context paragraph said the build standard "routes every gate through
`.cargo/config.toml`". Kani and Verus are not Cargo gates in that sense —
`make kani` invokes `$(CARGO) kani` and `make verus` invokes the pinned
`rust-prover-tools` package, each on the toolchain its own verifier release
requires — and Whitaker runs under its installer-managed toolchain. The
paragraph now says "every Cargo gate", matching the decision outcome that
already enumerated the exceptions.

The first consequence claimed "a gate result is always attributable to the
pinned stable toolchain". That is true of the Cargo gates and false of the
three verifiers, which is why the decision outcome lists them. It now says "a
result from a Cargo gate", and the exceptions are named as standing rather than
as something this decision introduces.

Neither correction changes the decision. What the record protects is still that
no gate selects Cranelift, and that none of the three exceptions is reachable
through the development fragment.

The same review found a gap in the decision's own machinery rather than in its
prose. The pin was read with `$(file <tools/dev-fast/TOOLCHAIN)`, which yields
nothing for a missing file and strips to nothing for an empty one, and every
recipe then used it as `cargo +$(DEV_FAST_TOOLCHAIN)` or
`rustup toolchain install $(DEV_FAST_TOOLCHAIN)`. An empty read therefore did
not fail: it fell back to the default toolchain, which is exactly the outcome
the pin exists to prevent, and the route would have reported success on a
toolchain it never selected.

Each development recipe now begins with `$(DEV_FAST_CHECK)`, which refuses an
empty read through `$(error …)` before any command runs. The check is
per-recipe rather than a file-wide `$(error)`, for the reason the host probe's
is: the standard gates parse `DEV_FAST_TOOLCHAIN` as well, and a broken
development pin must not take them down with it.
`tests/workflow_contracts/dev_fast_pin_guard_test.py` holds both halves, and
removes the guard in a private copy to show the defect returning without it.

Neither change alters the decision. What changes is that the pin can no longer
be silently skipped, and that both development targets are now exercised end to
end rather than only through dry runs.

## Addendum — 2026-11-13

The decision above describes `dev-test` as a convenience route that runs the
same suite on a different backend. One case is not run, and the exclusion
deserves recording here rather than only in the developers' guide, because it
is the single place where the development route is knowingly narrower than the
merge gate.

Cranelift aborts on a panic it cannot unwind
(`fatal runtime error: failed to initiate panic, error 5`). The regression
`hnsw::cpu::test_helpers::tests::graph_helpers_report_poisoning_without_reconfiguring`
induces a panic while a graph write guard is alive, then inspects the poison
through `catch_unwind`. Under the development backend the process dies before
the assertion is reached, so the case cannot report a meaningful result there.

`DEV_TEST_FILTER` therefore excludes exactly that case, named once through the
`POISONED_LOCK_TEST` variable so a rename is a one-line edit. The exclusion is
scoped to the development route only: `make test`, on the pinned stable
toolchain, keeps running the regression, and it is a merge gate. A contract
test requires the exclusion to name a declaration that actually exists at the
module level the path implies, because nextest treats an unmatched `test(=...)`
clause as an empty match and would otherwise silently run the aborting case.

This narrows the development route's coverage; it does not narrow the gate's.
The rule that no gate selects Cranelift is unchanged.
