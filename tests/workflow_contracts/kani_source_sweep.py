"""Find the sources a crate walk leaves behind.

A module walk reads what the crate roots reach. This module covers what they do
not: a ``mod`` declaration's child file, the proof-bearing sources nothing
declares, and the directory sweep that finds them without entering a build
tree. An orphaned proof would otherwise vanish from the inventory without a
report, which is the failure the manifest contract exists to catch.

Re-use policy: import from :mod:`kani_harness_reading` only. The sweep and the
walk together are one reader; a second consumer should extend this module
rather than grow another traversal.
"""

from __future__ import annotations

import re
import typing as typ
from pathlib import Path

from cargo_packages_reading import _Package, _is_package_tree
from kani_attribute_reading import _proof_attributes, _proof_kind
from kani_module_paths import _Location, _child_file
from kani_source_reading import _ModuleItem, _Source, _module_chain, _parse
from rust_source_reading import _FUNCTION


def _declared_children(
    location: _Location,
    source: _Source,
    module: tuple[str, ...],
) -> typ.Iterator[tuple[_ModuleItem, _Location | None, tuple[str, ...]]]:
    """Yield each file-based module declaration's item, target and path.

    ``child`` is None when the declaration names no readable file; the caller
    decides whether that is a report (a ``#[path]``) or Rust's own error (a
    plain ``mod name;``).

    A child whose enclosing inline module has an unreadable ``#[path]`` gets
    None too: that module's directory is unknown, so resolving the child
    against the module's name would look where Rust may not.
    """
    for item in source.modules:
        if item.inline:
            continue
        chain = _module_chain(source.modules, item.start)
        guessed = item.path_unreadable or any(
            parent.path_unreadable for parent in chain
        )
        child = (
            None
            if guessed
            else _child_file(location, chain, item.name, item.path_value)
        )
        child_location = (
            None
            if child is None
            else _Location(
                file=child,
                is_crate_root=False,
                via_path=item.path_value is not None,
            )
        )
        child_module = (*module, *(parent.name for parent in chain), item.name)
        yield item, child_location, child_module


def _holds_proof(source: _Source) -> bool:
    """Return whether `source` declares at least one proof-shaped function."""
    return any(
        _proof_kind(_proof_attributes(source.code, match.start())) is not None
        for match in _FUNCTION.finditer(source.code)
    )


def _unreached(packages: typ.Iterable[_Package], visited: frozenset[Path]) -> list[Path]:
    """Return proof-bearing sources under `packages` that no module reached.

    The sweep covers every crate root's source directories, not only ``src``
    and not only the first root. A package may hold a library at ``src`` and a
    binary elsewhere; a proof left beside the binary would otherwise be read by
    nothing and reported by nothing, so a declaration that stopped being
    reached would leave the inventory in silence.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(_sweep_files(packages)):
        resolved = path.resolve()
        if resolved in visited or resolved in seen:
            continue
        seen.add(resolved)
        source = _parse(path)
        if source is not None and _holds_proof(source):
            found.append(path)
    return found


def _sweep_files(packages: typ.Iterable[_Package]) -> typ.Iterator[Path]:
    """Yield each source file under the packages' own directories once."""
    candidates = [
        candidate
        for package in packages
        for candidate in (package.directory / "src", package.root.parent)
    ]
    for directory in sorted({candidate.resolve() for candidate in candidates if candidate.is_dir()}):
        yield from _source_files(directory)


def _source_files(directory: Path) -> typ.Iterator[Path]:
    """Yield each Rust file below `directory`, never entering an ignored tree.

    The sweep may start at a package directory when its library lives outside
    ``src``, and a build directory there holds generated sources. Reading those
    would report an orphan the repository never wrote.
    """
    seen: set[Path] = set()
    pending = [directory]
    while pending:
        current = pending.pop()
        if current.resolve() in seen:
            continue
        seen.add(current.resolve())
        entries = sorted(current.iterdir())
        pending.extend(entry for entry in entries if _is_package_tree(entry))
        yield from (entry for entry in entries if entry.suffix == ".rs")


