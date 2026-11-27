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
    declared = _declared_entries(document)
    explicit = [
        root
        for entry in declared
        for root in _declared_binary(directory, entry, _package_name(document))
    ]
    if _autobins_disabled(document):
        return [root for root in explicit if root.is_file()]
    return [*explicit, *_discovered_binaries(document, directory, declared, explicit)]


def _declared_entries(document: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return the package's ``[[bin]]`` tables."""
    return [entry for entry in document.get("bin") or [] if isinstance(entry, dict)]


def _package_name(document: dict[str, typ.Any]) -> object:
    """Return the package's declared name, if the document names one."""
    package = document.get("package")
    return package.get("name") if isinstance(package, dict) else None


def _discovered_binaries(
    document: dict[str, typ.Any],
    directory: Path,
    declared: list[dict[str, typ.Any]],
    explicit: list[Path],
) -> list[Path]:
    """Return the automatic binary roots no explicit target claims.

    Cargo does not build a ``src/bin`` file whose name an explicit ``[[bin]]``
    already declares; reading it anyway would count a proof the gate never
    compiles.
    """
    package_name = _package_name(document)
    claimed = {entry.get("name") for entry in declared if entry.get("name")}
    claimed_paths = {root.resolve() for root in explicit}
    automatic = [directory / "src" / "main.rs", *_automatic_binaries(directory)]
    return [
        root
        for root in automatic
        if _binary_name(root, package_name) not in claimed
        and root.resolve() not in claimed_paths
    ]


def _binary_name(root: Path, package_name: object) -> str:
    """Return the target name Cargo gives the automatic binary at `root`.

    ``src/main.rs`` is always the package's own target, whatever the package
    directory is called or however an explicit target spells its path.
    """
    if root.name != "main.rs":
        return root.stem
    if root.parent.name == "src" and isinstance(package_name, str):
        return package_name
    return root.parent.name


def _declared_binary(
    directory: Path, entry: dict[str, typ.Any], package_name: object
) -> list[Path]:
    """Return the path one ``[[bin]]`` entry names, explicit or inferred.

    Cargo resolves a declaration that gives only a name to ``src/main.rs``
    when the name is the package's own, and to ``src/bin/<name>.rs`` (or
    ``src/bin/<name>/main.rs``) otherwise. That resolution happens whether or
    not auto-discovery is enabled, because the declaration is the caller's.
    """
    if isinstance(entry.get("path"), str):
        return [directory / entry["path"]]
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        return []
    binaries = directory / "src" / "bin"
    candidates = [binaries / f"{name}.rs", binaries / name / "main.rs"]
    if name == package_name:
        candidates.insert(0, directory / "src" / "main.rs")
    existing = [candidate for candidate in candidates if candidate.is_file()]
    return existing[:1] or candidates[:1]


def _autobins_disabled(document: dict[str, typ.Any]) -> bool:
    """Return whether the package turns Cargo's binary discovery off."""
    package = document.get("package")
    autobins = package.get("autobins") if isinstance(package, dict) else None
    return autobins is False


def _automatic_binaries(directory: Path) -> list[Path]:
    """Return the binary targets Cargo discovers without a declaration."""
    binaries = directory / "src" / "bin"
    if not binaries.is_dir():
        return []
    return [*sorted(binaries.glob("*.rs")), *sorted(binaries.glob("*/main.rs"))]


