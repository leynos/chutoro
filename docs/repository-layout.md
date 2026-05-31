# Repository layout

This document explains the repository tree for Chutoro contributors. Use it
alongside the user's guide, developer's guide, and design document when deciding
where a change belongs. The tree below is intentionally compact and omits
generated artefacts such as `target/`.

```plaintext
.
├── .cargo/
├── .codescene/
├── .config/
├── .github/
├── chutoro-benches/
├── chutoro-cli/
├── chutoro-core/
├── chutoro-providers/
├── chutoro-test-support/
├── docs/
├── scripts/
├── tools/
├── verus/
├── AGENTS.md
├── Cargo.lock
├── Cargo.toml
├── Makefile
├── README.md
└── rust-toolchain.toml
```

_Figure 1: Compact repository tree for contributor orientation._

## Top-level files

| Path                  | Responsibility                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `AGENTS.md`           | Normative repository instructions for automation agents and maintainers using those workflows. |
| `Cargo.toml`          | Workspace manifest and shared Cargo configuration for the Rust crates.                         |
| `Cargo.lock`          | Locked dependency graph for reproducible workspace builds.                                     |
| `Makefile`            | Canonical command entrypoint for build, format, lint, test, proof, and documentation gates.    |
| `README.md`           | Project overview and public entrypoint.                                                        |
| `rust-toolchain.toml` | Rust toolchain selection for the workspace.                                                    |

_Table 1: Top-level file responsibilities._

## Source crates

| Path                    | Responsibility                                                        |
| ----------------------- | --------------------------------------------------------------------- |
| `chutoro-core/`         | Core library implementation and its crate-local tests.                |
| `chutoro-cli/`          | Command-line application crate and user-facing command orchestration. |
| `chutoro-providers/`    | Provider implementations grouped by provider family.                  |
| `chutoro-benches/`      | Benchmark crate, benchmark harnesses, and benchmark support code.     |
| `chutoro-test-support/` | Shared test utilities used by workspace tests.                        |

_Table 2: Workspace crate responsibilities._

## Documentation and planning

| Path                                | Responsibility                                              |
| ----------------------------------- | ----------------------------------------------------------- |
| `docs/contents.md`                  | Canonical index for repository documentation.               |
| `docs/documentation-style-guide.md` | Documentation style, document-type, and formatting rules.   |
| `docs/chutoro-design.md`            | Primary system design reference.                            |
| `docs/developers-guide.md`          | Maintainer workflows and development guidance.              |
| `docs/users-guide.md`               | User-facing operation and integration guidance.             |
| `docs/roadmap.md`                   | Roadmap phases, steps, tasks, and acceptance criteria.      |
| `docs/execplans/`                   | Living execution plans for substantial implementation work. |

_Table 3: Documentation and planning responsibilities._

## Tooling and verification

| Path                   | Responsibility                                                         |
| ---------------------- | ---------------------------------------------------------------------- |
| `.cargo/`              | Workspace-local Cargo configuration.                                   |
| `.config/nextest.toml` | `cargo-nextest` profiles and test-runner configuration.                |
| `.github/`             | GitHub workflows, Dependabot configuration, and repository automation. |
| `.codescene/`          | CodeScene code health rule configuration.                              |
| `scripts/`             | Maintainer scripts for proof tooling and operational support.          |
| `tools/`               | Tool-specific support files that do not belong to a Rust crate.        |
| `verus/`               | Verus proof sources for formally verified properties.                  |

_Table 4: Tooling and verification path responsibilities._

## Generated artefacts

Cargo build output belongs under `target/` and must not be committed. Temporary
command logs should be written under `/tmp` using a branch-specific filename,
as described in `AGENTS.md`, rather than being added to the repository.

Generated tooling directories must not be edited by hand:

- `target/` is generated Cargo build output.
- `.verus/` is generated Verus/tooling output.
- `.memdb/` is generated local code-graph state.

## Test locations

- Unit tests usually live next to the code under each crate's `src/` tree.
- Behavioural and integration tests live under crate-local `tests/`
  directories, such as `chutoro-core/tests/`.
- Feature files for BDD tests live under `tests/features/` in the crate that
  owns the behaviour.
- Compile-surface tests use dedicated integration-test binaries with fixtures
  under the relevant crate's `tests/trybuild/` directory, such as
  `chutoro-core/tests/trybuild/`.

## Core session module

Session code is split by responsibility under `chutoro-core/src/session/`:

- `mod.rs` owns the `ClusteringSession` struct, lightweight read-only
  accessors, public re-exports, and the high-level Rustdoc contract.
- `config.rs` owns `SessionRefreshPolicy` and `SessionConfig`, the small value
  types carried by each session.
- `session_impl.rs` owns construction, `append`, HNSW error mapping, and the
  edge-harvesting write path.
- `core_distance.rs` owns the pure core-distance helpers and the recompute
  workflow. The pure helpers must not depend on HNSW adapter internals beyond
  the public `Neighbour` value.
- `clock.rs` is compiled only with the `metrics` feature and owns the
  monotonic-clock support used for deterministic latency tests.
