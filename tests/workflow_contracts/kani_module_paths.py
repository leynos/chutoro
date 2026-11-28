"""Resolve where Rust looks for a module's children.

Rust's module-file rule is not one rule but several: a crate root and a
``mod.rs`` resolve children from their own directory, while ``src/outer.rs``
resolves them from ``src/outer/``. A ``#[path]`` attribute replaces that
directory for the declaration it sits on, and an inline module contributes a
directory of its own. Getting one of these wrong sends the reader looking for a
file Rust never loads, and the proof it holds is then reported unreachable.

Re-use policy: import from :mod:`kani_harness_reading` only. The rules are
rustc's, so a second consumer should extend this module rather than restate
them.
"""

from __future__ import annotations

import typing as typ
from pathlib import Path

from kani_source_reading import _ModuleItem


class _Location(typ.NamedTuple):
    """Where a file sits in the module tree, for resolving its children.

    Attributes
    ----------
    file
        The file being read.
    is_crate_root
        Whether it is a crate root.
    via_path
        Whether a ``#[path]`` attribute named it. Rust gives such a file the
        same directory rule as ``mod.rs``: its children resolve from the
        file's own parent directory, not from a directory named after its
        stem. Without that, a ``#[path]``-loaded module's children would be
        looked for in a directory Rust never uses, and their proofs would be
        reported unreachable.
    """

    file: Path
    is_crate_root: bool
    via_path: bool


class _Context(typ.NamedTuple):
    """The state one file is read in.

    Attributes
    ----------
    module
        The module path the file is reached by.
    guards
        The ``#[cfg]`` attributes of the declarations that reached it.
    binary
        Whether the file belongs to a binary target, which ``--lib`` never
        compiles.
    """

    module: tuple[str, ...]
    guards: tuple[str, ...]
    binary: bool


def _is_mod_rs(location: _Location) -> bool:
    """Report whether `location` follows the ``mod.rs`` directory rule."""
    return (
        location.is_crate_root
        or location.via_path
        or location.file.name == "mod.rs"
    )


def _child_directory(location: _Location) -> Path:
    """Return the directory in which `location` resolves its child modules."""
    if _is_mod_rs(location):
        return location.file.parent
    return location.file.with_suffix("")


def _inline_directory(item: _ModuleItem) -> str:
    """Return the directory name an inline module contributes to its children.

    Normally that is the module's own name. A ``#[path]`` on the inline module
    replaces the directory instead: for
    ``#[path = "thread_files"] mod thread { #[path = "tls.rs"] mod local_data; }``,
    Rust loads ``thread_files/tls.rs``, so the module contributes
    ``thread_files`` and keeps ``thread`` only for its harness identity.
    """
    if item.path_value is None:
        return item.name
    return item.path_value.rstrip("/")


def _path_base(location: _Location, chain: tuple[_ModuleItem, ...]) -> Path:
    """Return the directory a ``#[path]`` value is relative to.

    Rust resolves a path attribute on a declaration outside an inline module
    block relative to the containing file's directory, and one inside an
    inline block relative to the module directory with the inline blocks as
    directories. An inline block's own ``#[path]`` renames that directory.
    """
    if not chain:
        return location.file.parent
    return _inline_base(location, chain)


def _inline_base(location: _Location, chain: tuple[_ModuleItem, ...]) -> Path:
    """Return the directory an inline module chain resolves its files from.

    A file module's children normally live under a directory named after the
    file's stem (``src/outer.rs`` gives ``src/outer/``). A ``#[path]`` on the
    first inline block replaces that directory rather than nesting inside it:
    ``#[path = "tf"] mod thread { mod inner; }`` in ``src/outer.rs`` reads
    ``src/tf/inner.rs``, and ``mod a { #[path = "tf"] mod thread { .. } }``
    reads ``src/outer/a/tf/inner.rs``. The stem therefore applies only when
    the first block carries no path of its own.
    """
    base = (
        location.file.parent
        if chain and chain[0].path_value is not None
        else _child_directory(location)
    )
    return base.joinpath(*(_inline_directory(item) for item in chain))


def _child_file(
    location: _Location,
    chain: tuple[_ModuleItem, ...],
    name: str,
    path_value: str | None,
) -> Path | None:
    """Return the file a ``mod name;`` declaration reads, if one exists."""
    if path_value is not None:
        candidate = _path_base(location, chain) / path_value
        return candidate if candidate.is_file() else None
    directory = _inline_base(location, chain)
    for candidate in (directory / f"{name}.rs", directory / name / "mod.rs"):
        if candidate.is_file():
            return candidate
    return None


