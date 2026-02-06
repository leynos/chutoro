# Verus toolchain

The edge harvest proofs use Verus and must run with a pinned toolchain to keep
continuous integration (CI) deterministic.

## Pinned version

The repository pins Verus to the release listed in `tools/verus/VERSION`:
`0.2026.01.30.44ebdee`. Use the matching asset from the
[Verus releases page](https://github.com/verus-lang/verus/releases).

## Install (recommended)

Use the helper script, which reads the pinned version file, downloads the
matching release asset for Linux, and verifies its SHA-256 checksum:

```bash
scripts/install-verus.sh
```

Expected checksums live in `tools/verus/SHA256SUMS` and should be updated when
the Verus version changes.

Optional environment variables:

- `VERUS_INSTALL_DIR` to override the installation location.
- `VERUS_TARGET` to select a different release asset (for example
  `x86-macos`).

After installation, export the binary path:

```bash
export VERUS_BIN="/path/to/verus/verus"
```

## Run proofs

The make target is self-contained and downloads Verus with the required Rust
toolchain installation when missing. It respects `VERUS_BIN`, which may point
at a Verus binary or installation directory:

```bash
make verus
```

## Rust toolchain

The Verus binary reports the exact Rust toolchain it needs. The `make verus`
target parses the toolchain from `verus --version` and installs it via `rustup`
when missing.
