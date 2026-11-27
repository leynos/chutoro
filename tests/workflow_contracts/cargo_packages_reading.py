"""Find the workspace's packages and assemble their crate roots.

This module answers which ``Cargo.toml`` files the workspace holds and, for
each, which crate roots to read. What a manifest's roots *are* -- its library
and binary targets -- is :mod:`cargo_targets_reading`'s question; reading the
Rust source behind them is :mod:`kani_harness_reading`'s.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import tomllib
import typing as typ
from pathlib import Path

from cargo_targets_reading import (
    _Package,
    _binary_roots,
    _declared_roots,
    _library_root,
)
from kani_source_reading import IGNORED_DIRECTORIES

def _manifests(root: Path) -> list[Path]:
    """Return every package manifest under `root`, never entering an ignored tree.

    Pruning during the walk rather than filtering afterwards is what keeps a
    `target` tree from being traversed at all: a compiled workspace can hold
    tens of thousands of `Cargo.toml` files under one, and reading them only to
    discard each by path would make the sweep quadratic in the wrong thing.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        resolved = directory.resolve()
        # A directory symlink may point at an ancestor, so the same tree could
        # otherwise be queued again under an ever-longer path.
        if resolved in seen:
            continue
        seen.add(resolved)
        entries = sorted(directory.iterdir())
        found.extend(entry for entry in entries if _is_manifest(entry))
        pending.extend(entry for entry in entries if _is_package_tree(entry))
    return sorted(found)


def _is_manifest(path: Path) -> bool:
    """Return whether `path` is a package manifest."""
    return path.name == "Cargo.toml" and path.is_file()


def _is_package_tree(path: Path) -> bool:
    """Return whether `path` is a directory the sweep may enter."""
    return path.is_dir() and path.name not in IGNORED_DIRECTORIES


class _ManifestPackage(typ.NamedTuple):
    """One package manifest and the crate roots it declares.

    Attributes
    ----------
    name
        The package name.
    directory
        The package directory.
    roots
        The crate roots to read, in declaration order: an explicit
        ``[lib] path`` or the default ``src/lib.rs``, then the binary roots.
    library_roots
        Those of `roots` that are library targets, which ``--lib`` compiles
        and the gate therefore verifies.
    """

    name: str
    directory: Path
    roots: tuple[Path, ...]
    library_roots: tuple[Path, ...]


def _package_directories(root: Path) -> list[_ManifestPackage]:
    """Return each package under `root` and the roots it declares."""
    found: list[_ManifestPackage] = []
    for manifest in _manifests(root):
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        package = document.get("package")
        if not isinstance(package, dict):
            continue
        roots = _declared_roots(document, manifest.parent)
        if any(candidate.is_file() for candidate in roots):
            library = _library_root(document, manifest.parent)
            found.append(
                _ManifestPackage(
                    name=package["name"],
                    directory=manifest.parent,
                    roots=tuple(
                        candidate for candidate in roots if candidate.is_file()
                    ),
                    library_roots=(
                        (library,)
                        if library is not None and library.is_file()
                        else ()
                    ),
                )
            )
    return found


def _packages(root: Path) -> list[_Package]:
    """Return each package whose crate root can be read."""
    packages: list[_Package] = []
    for package in _package_directories(root):
        for crate_root in package.roots:
            packages.append(
                _Package(
                    name=package.name,
                    root=crate_root,
                    directory=package.directory,
                    is_binary=crate_root not in package.library_roots,
                )
            )
    return packages
