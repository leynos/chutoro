"""Fixture lockfiles must resolve only versions the workspace also locks.

The compile-contract fixtures are separate workspaces, but their checks run
``cargo check --offline --locked`` after CI has fetched only the workspace
lockfile. Any fixture package version absent from the workspace lock is
therefore unavailable offline, which is how ``bytes 1.12.2`` once failed the
``staging_lock`` fixture while the workspace pinned ``1.12.1``.

Seed each fixture lockfile from the workspace one, then let Cargo prune it:

```plaintext
cp Cargo.lock chutoro-compile-contracts/tests/fixtures/<name>/Cargo.lock
cargo metadata --format-version 1 --offline \
  --manifest-path chutoro-compile-contracts/tests/fixtures/<name>/Cargo.toml
```
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "chutoro-compile-contracts" / "tests" / "fixtures"


def locked_versions(lockfile: Path) -> dict[str, set[str]]:
    """Return every package version named by one ``Cargo.lock``.

    Parameters
    ----------
    lockfile : pathlib.Path
        Lockfile to read.

    Returns
    -------
    dict[str, set[str]]
        Package name to the set of versions the lockfile names.
    """
    data = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    versions: dict[str, set[str]] = {}
    for package in data["package"]:
        versions.setdefault(package["name"], set()).add(package["version"])
    return versions


def fixture_name(lockfile: Path) -> str:
    """Return the package name Cargo gives a fixture directory.

    Parameters
    ----------
    lockfile : pathlib.Path
        A fixture's ``Cargo.lock``.

    Returns
    -------
    str
        The directory name with underscores replaced by hyphens.
    """
    return lockfile.parent.name.replace("_", "-")


def test_fixture_locks_stay_within_the_workspace_lock() -> None:
    """Every fixture package version must exist in the workspace lockfile."""
    workspace = locked_versions(ROOT / "Cargo.lock")
    failures: list[str] = []
    for lockfile in sorted(FIXTURES.glob("*/Cargo.lock")):
        own = fixture_name(lockfile)
        for name, versions in sorted(locked_versions(lockfile).items()):
            if name == own:
                continue
            unknown = versions - workspace.get(name, set())
            if unknown:
                failures.append(
                    f"{lockfile.relative_to(ROOT)}: {name} "
                    f"{sorted(unknown)} is not in the workspace lock"
                )
    assert not failures, "\n".join(
        ["fixture lockfiles must be seeded from the workspace lock:", *failures]
    )