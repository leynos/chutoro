# Verus Toolchain

The edge harvest proofs use Verus and must run with a pinned toolchain to keep
CI deterministic.

## Pinned Version

The repository pins Verus to the release listed in `tools/verus/VERSION`:
`0.2026.01.30.44ebdee`. Use the matching asset from the
[Verus releases page](https://github.com/verus-lang/verus/releases).

## Install (Recommended)

Use the helper script, which reads the pinned version file and downloads the
matching release asset for Linux:

```bash
scripts/install-verus.sh
```

Optional environment variables:

- `VERUS_INSTALL_DIR` to override the installation location.
- `VERUS_TARGET` to select a different release asset (for example
  `x86-macos`).

After installation, export the binary path:

```bash
export VERUS_BIN="/path/to/verus/verus"
```

## Run Proofs

Use the make target, which is self-contained and will download Verus and
install the required Rust toolchain if they are missing. It respects
`VERUS_BIN` if you want to point at a custom install:

```bash
make verus
```

## Rust Toolchain

The Verus binary reports the exact Rust toolchain it needs. The `make verus`
target parses the toolchain from `verus --version` and installs it via `rustup`
when missing.
