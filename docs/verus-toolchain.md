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

Use the make target, which respects `VERUS_BIN`:

```bash
make verus
```

## Rust Toolchain

The Verus binary will report the exact Rust toolchain it needs. Follow the
prompted `rustup` instructions if the toolchain is missing.
