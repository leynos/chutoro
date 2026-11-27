"""Discover each Cargo package and the crate roots it declares.

This module answers only what Cargo builds: which ``Cargo.toml`` files the
workspace holds, which library and binary targets each declares, and which of
those targets are library roots -- the ones ``--lib`` compiles and the gate
therefore verifies. Reading the Rust source behind those roots, and attributing
harnesses to a module path, is :mod:`kani_harness_reading`'s work.

Keeping the two apart gives workspace and target-layout changes one test
surface, and keeps the source walk from having to know how a package is
published or which binaries it auto-discovers.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import typing as typ
from pathlib import Path

class _Package(typ.NamedTuple):
    """A workspace member whose library or binary crate can hold proofs.

    Attributes
    ----------
    name
        The package name as ``Cargo.toml`` declares it.
    root
        The crate root source file, ``src/lib.rs`` or ``src/main.rs``.
    directory
        The package directory, used to find sources no module reaches.
    is_binary
        Whether the root is a binary target, which `--lib` never compiles.
    """

    name: str
    root: Path
    directory: Path
    is_binary: bool



def _declared_roots(
    document: dict[str, typ.Any], directory: Path
) -> tuple[Path, ...]:
    """Return the crate roots a manifest declares.

    Cargo lets a library live wherever ``[lib] path`` points, so a package with
    no ``src`` directory at all is still a package whose proofs compile. Its
    explicit path is honoured; otherwise the conventional ``src/lib.rs`` is
    used. Binaries are read too: an explicit ``[[bin]] path``, the
    conventional ``src/main.rs``, and each file under ``src/bin/``. Skipping
    any of them would omit a package's proofs without a report, which is the
    failure the inventory contract exists to prevent.
    """
    roots: list[Path] = []
    library = _library_root(document, directory)
    if library is not None:
        roots.append(library)
    roots.extend(_binary_roots(document, directory))
    return tuple(roots)


def _library_root(document: dict[str, typ.Any], directory: Path) -> Path | None:
    """Return the manifest's library root, explicit or conventional.

    ``autolib = false`` tells Cargo not to build ``src/lib.rs`` as a library of
    its own, usually because a binary loads that file as a module. Inferring it
    anyway would visit the same source as two crate roots and report the
    second visit as a repeat that is not there. An explicit ``[lib] path`` is
    still honoured, because that key asks for a library.
    """
    library = document.get("lib")
    if isinstance(library, dict):
        if isinstance(library.get("path"), str):
            return directory / library["path"]
        return directory / "src" / "lib.rs"
    package = document.get("package")
    autolib = package.get("autolib") if isinstance(package, dict) else None
    if autolib is False:
        return None
    return directory / "src" / "lib.rs"


def _binary_roots(document: dict[str, typ.Any], directory: Path) -> list[Path]:
    """Return every binary crate root the manifest implies.

    An explicit ``[[bin]]`` may name its own path; a package may also declare
    no binaries at all, in which case Cargo discovers ``src/main.rs`` and each
    ``src/bin/*.rs``. Both are read, because a proof in a binary target is
    compiled exactly as one in a library is.

    A package that sets ``autobins = false`` turns discovery off, usually
    because a library loads those files as modules on purpose. Reading them
    again as binaries would visit the same source twice and report a
    ``repeated-source`` that is not there.
    """
    declared = document.get("bin")
    explicit = [
        directory / entry["path"]
        for entry in declared or []
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    ]
    package = document.get("package")
    autobins = package.get("autobins") if isinstance(package, dict) else None
    if autobins is False:
        return [root for root in explicit if root.is_file()]
    discovered = [directory / "src" / "main.rs"]
    binaries = directory / "src" / "bin"
    if binaries.is_dir():
        discovered.extend(sorted(binaries.glob("*.rs")))
        discovered.extend(sorted(binaries.glob("*/main.rs")))
    return [*explicit, *(root for root in discovered if root.is_file())]


