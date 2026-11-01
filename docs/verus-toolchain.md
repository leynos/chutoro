# Verus toolchain

The edge harvest proofs use Verus and must run with a pinned toolchain to keep
continuous integration (CI) deterministic.

## Pinned version

The repository pins Verus to the release listed in `tools/verus/VERSION`:
`0.2026.01.30.44ebdee`. Use the matching asset from the
[Verus releases page](https://github.com/verus-lang/verus/releases).

## Install (recommended)

Use the Make target, which delegates to
[`rust-prover-tools`](https://github.com/leynos/rust-prover-tools). That
package reads the pinned version file, downloads the matching release asset for
Linux over a bounded HTTPS connection, and verifies its SHA-256 checksum before
extraction:

```bash
make install-verus
```

Expected checksums live in `tools/verus/SHA256SUMS` and should be updated when
the Verus version changes. The package reads both files by default, so the pins
stay in this repository.

Optional flags and their environment bindings:

- `--install-dir` or `VERUS_INSTALL_DIR` to override the installation location.
- `--target` or `VERUS_TARGET` to select a different release asset (for example
  `x86-macos`).

After installation, export the binary path:

```bash
export VERUS_BIN="/path/to/verus/verus"
```

## Run proofs

The make target delegates to `prover-tools verus run`, which installs Verus and
its required Rust toolchain when either is missing. It respects `VERUS_BIN`,
which may point at a Verus binary or installation directory. Verus accepts one
input filename per invocation, so the target passes each proof file in turn and
stops at the first failure:

```bash
make verus
```

`VERUS_PROOF_FILES` overrides the default proof list.

## Rust toolchain

The Verus binary reports the exact Rust toolchain it needs. The `make verus`
target, through the shared package, parses the toolchain from `verus --version`
and installs it via `rustup` when missing.
