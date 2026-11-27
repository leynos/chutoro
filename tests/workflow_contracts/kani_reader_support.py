"""Shared fixtures for the Kani source reader's contract tests.

Both reader-contract modules build tiny packages and ask :func:`discover` what
it makes of them, so the package writer and the two inventory summaries live
here rather than in either module.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

from pathlib import Path

from kani_harness_reading import Inventory

#: A package manifest, enough for the sweep to recognise a package.
PACKAGE_MANIFEST = '[package]\nname = "{name}"\nversion = "0.1.0"\n'


def _write_package(root: Path, source: str, name: str = "fixture") -> Path:
    """Create a one-file package under `root` and return its crate root."""
    directory = root / name
    (directory / "src").mkdir(parents=True)
    (directory / "Cargo.toml").write_text(
        PACKAGE_MANIFEST.format(name=name), encoding="utf-8"
    )
    crate_root = directory / "src" / "lib.rs"
    crate_root.write_text(source, encoding="utf-8")
    return crate_root


def _names(inventory: Inventory) -> list[str]:
    """Return the harness names the inventory found, sorted."""
    return sorted(proof.pretty_name for proof in inventory.proofs)


def _kinds(inventory: Inventory) -> list[str]:
    """Return the kinds of unresolved report, sorted."""
    return sorted(report.kind for report in inventory.unresolved)
