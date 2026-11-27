"""Read every Kani proof declaration in the workspace tree.

The workflow-contract suite needs the whole proof surface, not merely the
files whose names happen to match ``kani_*``. Six proofs live in an inline
``#[cfg(kani)] mod kani_proofs`` block at the bottom of
``chutoro-core/src/distance/mod.rs``, and a filename-based inventory never saw
them. This module therefore walks each crate's module tree and reads
declarations out of the source.

It reuses :mod:`kani_source_reading` for the per-file reading: that module
knows where code stops and a comment, string or macro body begins, and it
derives the inline module chain enclosing a declaration. This module adds the
file-based link: a module reached through ``mod name;`` contributes its
declaration's name, and a ``#[path]`` that names no readable file is reported
rather than skipped.

Anything this reader cannot attribute to a module path is reported rather than
skipped: an unresolvable ``#[path]`` module, a proof wrapped in ``cfg_attr``, a
proof inside a macro definition, and a proof-bearing file no ``mod``
declaration reaches are all unsupported shapes, and the contract turns each
report into a failure. Failing closed is the point; a reader that guessed
would let a proof disappear from the inventory without anything noticing.

Re-use policy: import from :mod:`kani_manifest_test` and the rewritten
:mod:`kani_fast_tier_test` only. Do not grow a second Rust scanner beside it.
"""

from __future__ import annotations

import dataclasses
import typing as typ
from pathlib import Path

from rust_source_reading import _FUNCTION

from cargo_packages_reading import _Package, _is_package_tree, _packages
from kani_source_reading import (
    _ModuleItem,
    _Source,
    _gated_by_test,
    _inside,
    _line_of,
    _module_chain,
    _parse,
    _proof_attributes,
    _proof_kind,
)
from workflow_support import ROOT


@dataclasses.dataclass(frozen=True)
class ProofDeclaration:
    """One proof declaration and the identity Kani will print for it.

    Attributes
    ----------
    package
        The Cargo package the declaration compiles into.
    pretty_name
        The fully qualified harness path, which is both Kani's
        ``Checking harness <name>`` text and its ``--harness --exact`` selector.
    path
        The repository-relative source file holding the declaration.
    line
        The one-based line of the proof function's declaration in that file.
    """

    package: str
    pretty_name: str
    path: str
    line: int


@dataclasses.dataclass(frozen=True)
class UnresolvedDeclaration:
    """A proof-shaped construct this reader cannot attribute to a path.

    Attributes
    ----------
    kind
        ``path-attribute``, ``cfg-attr``, ``macro`` or ``unreachable``; the
        reason the declaration could not be read.
    path
        The repository-relative source file carrying the construct.
    line
        The one-based line of the construct.
    detail
        A human-readable explanation naming what was found.
    """

    kind: str
    path: str
    line: int
    detail: str


@dataclasses.dataclass(frozen=True)
class Inventory:
    """The proof surface read from one tree.

    Attributes
    ----------
    proofs
        Every attributable declaration, sorted by package and harness path.
    unresolved
        Every construct that could not be attributed, in source order.
    """

    proofs: tuple[ProofDeclaration, ...]
    unresolved: tuple[UnresolvedDeclaration, ...]


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


class _Reader:
    """Reads one package's module tree and records what it finds."""

    def __init__(self, package: str, repo_root: Path) -> None:
        """Start a sweep of one package under `repo_root`."""
        self._package = package
        self._repo_root = repo_root
        self._visited: dict[Path, tuple[str, ...]] = {}
        self.proofs: list[ProofDeclaration] = []
        self.unresolved: list[UnresolvedDeclaration] = []

    @property
    def visited(self) -> frozenset[Path]:
        """Return the resolved paths this sweep read."""
        return frozenset(self._visited)

    def _report(self, kind: str, path: str, line: int, detail: str) -> None:
        """Record one construct that could not be attributed."""
        self.unresolved.append(
            UnresolvedDeclaration(kind=kind, path=path, line=line, detail=detail)
        )

    def visit(
        self,
        file: Path,
        context: _Context = _Context((), (), False),
        *,
        is_crate_root: bool = False,
        via_path: bool = False,
    ) -> None:
        """Read `file` as the module `module`, following its declarations.

        `context` carries the module path, the ``#[cfg]`` attributes of the
        declarations that reached this file, and whether it belongs to a
        binary target. A module gated by a test guard puts every proof below it
        in that configuration too, and ``--lib`` never compiles a binary.

        A file reached twice under the same module path is a revisited module
        and is skipped. A file reached twice under *different* paths is not:
        Rust admits it when two declarations share a ``#[path]`` target, and
        each path yields its own harness identities. Dropping the second would
        silently under-report the inventory, so it is reported instead.
        """
        resolved = file.resolve()
        location = _Location(
            file=file, is_crate_root=is_crate_root, via_path=via_path
        )
        if resolved in self._visited:
            self._note_revisit(resolved, context, location)
            return
        self._visited[resolved] = context.module
        source = _parse(file)
        relative = file.relative_to(self._repo_root).as_posix()
        if source is None:
            # A file the sweep reached but could not decode is not the same as
            # one with no declarations: skipping it would shrink the inventory
            # silently, which is exactly the loss this contract exists to catch.
            self._report(
                "unreadable",
                relative,
                1,
                "the module resolves to this source, but it cannot be read as "
                "UTF-8, so its declarations cannot be checked",
            )
            return
        if source.inner:
            context = _Context(
                context.module,
                (*context.guards, *source.inner),
                context.binary,
            )
        self._record_proofs(source, relative, context)
        self._follow(location, source, context)

    def _note_revisit(
        self, resolved: Path, context: _Context, location: _Location
    ) -> None:
        """Report a source reached again under a different module path.

        Only a proof-bearing file can lose an identity by being read once.
        Sharing a test-only source between two ``#[path]`` modules is ordinary
        Rust and declares no harness, so reporting it would fail the contract
        on a valid tree.

        A proof-free file can still reach one, though: it may declare a child
        module, and that child's harnesses are read only under the first path.
        A repeat is therefore reported when either the file or a descendant of
        it declares a proof.
        """
        first = self._visited[resolved]
        if first == context.module:
            return
        source = _parse(location.file)
        if source is None:
            return
        held = _holds_proof(source) or bool(
            self._descendant_proofs(location, source, context)
        )
        if not held:
            return
        self._report(
            "repeated-source",
            location.file.relative_to(self._repo_root).as_posix(),
            1,
            "the same source is reached as both "
            f"{'::'.join(first)} and {'::'.join(context.module)}, so a harness "
            "identity below it would be dropped",
        )

    def _descendant_proofs(
        self, location: _Location, source: _Source, context: _Context
    ) -> list[str]:
        """Return the harness paths a repeated file's subtree would declare.

        The walk keeps its own set of resolved files rather than using
        :meth:`visit`, because it must terminate on the cyclic ``#[path]``
        graph Rust admits: a file that reaches itself would otherwise be
        followed forever under a module path that grows by one segment each
        time.
        """
        found: list[str] = []
        seen = {location.file.resolve()}
        pending = [(location, source, context.module)]
        while pending:
            current, current_source, current_module = pending.pop()
            for _, child, child_module in _declared_children(
                current, current_source, current_module
            ):
                if child is None or child.file.resolve() in seen:
                    continue
                seen.add(child.file.resolve())
                child_source = _parse(child.file)
                if child_source is None:
                    continue
                if _holds_proof(child_source):
                    found.append("::".join(child_module))
                    continue
                pending.append((child, child_source, child_module))
        return found

    def _record_proofs(
        self, source: _Source, relative: str, context: _Context
    ) -> None:
        """Record each proof in one file, or report why it cannot be."""
        for match in _FUNCTION.finditer(source.code):
            attributes = _proof_attributes(source.code, match.start())
            kind = _proof_kind(attributes)
            if kind is None:
                continue
            line = _line_of(source.text, match.start())
            if kind == "cfg-attr":
                self._report(
                    "cfg-attr",
                    relative,
                    line,
                    "proof is wrapped in cfg_attr, so whether it is a "
                    "declaration depends on configuration",
                )
                continue
            if _inside(match.start(), source.macros):
                self._report(
                    "macro",
                    relative,
                    line,
                    "proof is declared inside a macro definition",
                )
                continue
            chain = _module_chain(source.modules, match.start())
            # The gate compiles the library without `cfg(test)`, so a proof
            # behind a test guard exists in the source but never runs. Counting
            # it as a harness would let the manifest require something the
            # recipe cannot verify.
            guards = [
                *context.guards,
                *attributes,
                *(attr for item in chain for attr in item.attributes),
            ]
            if _gated_by_test(guards):
                self._report(
                    "cfg-test",
                    relative,
                    line,
                    "proof is behind a cfg(test) guard, which the Kani gate "
                    "does not enable, so it cannot run",
                )
                continue
            # The gate compiles with `--lib`, so a binary target's harnesses
            # never run in either tier. Recording one would let a mandatory
            # manifest entry be satisfied by a proof the run cannot execute.
            if context.binary:
                self._report(
                    "binary-target",
                    relative,
                    line,
                    "proof is declared in a binary target, which the Kani gate "
                    "does not compile",
                )
                continue
            pretty = "::".join(
                (*context.module, *(item.name for item in chain), match["name"])
            )
            self.proofs.append(
                ProofDeclaration(
                    package=self._package,
                    pretty_name=pretty,
                    path=relative,
                    line=line,
                )
            )

    def _follow(
        self, location: _Location, source: _Source, context: _Context
    ) -> None:
        """Follow each file-based module declaration in one file."""
        self._report_inline_paths(location, source)
        for item, child, child_module in _declared_children(
            location, source, context.module
        ):
            if child is None:
                self._report_missing_path(location, source, item)
                continue
            chain = _module_chain(source.modules, item.start)
            self.visit(
                child.file,
                _Context(
                    child_module,
                    (
                        *context.guards,
                        *(attr for parent in chain for attr in parent.attributes),
                        *item.attributes,
                    ),
                    context.binary,
                ),
                via_path=child.via_path,
            )

    def _report_inline_paths(self, location: _Location, source: _Source) -> None:
        """Report an inline module whose own ``#[path]`` cannot be read.

        That value renames the directory its children use, so a child resolved
        without it would be looked for where Rust may not put it.
        """
        for item in source.modules:
            if item.inline and item.path_unreadable:
                self._report_missing_path(location, source, item)

    def _report_missing_path(
        self, location: _Location, source: _Source, item: _ModuleItem
    ) -> None:
        """Report a ``#[path]`` declaration this reader cannot follow.

        A plain ``mod name;`` that names no file is Rust's own compile error,
        so only a ``#[path]`` is this reader's to explain. A value in a shape
        the reader cannot read is reported too: replacing it with the default
        rule would look for the module where the attribute does not point.
        """
        if item.path_unreadable:
            self._report(
                "path-attribute",
                location.file.relative_to(self._repo_root).as_posix(),
                _line_of(source.text, item.start),
                "the #[path] value is in a form this reader cannot read, so "
                "the module's file cannot be located",
            )
            return
        if item.path_value is None:
            return
        self._report(
            "path-attribute",
            location.file.relative_to(self._repo_root).as_posix(),
            _line_of(source.text, item.start),
            f"#[path = {item.path_value!r}] does not resolve to a readable "
            "source file",
        )


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


    """Return whether `source` declares at least one proof-shaped function."""
    return any(
        _proof_kind(_proof_attributes(source.code, match.start())) is not None
        for match in _FUNCTION.finditer(source.code)
    )


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


def discover(root: Path = ROOT) -> Inventory:
    """Return every proof declaration under `root` and every gap in reading it.

    Parameters
    ----------
    root
        The repository tree to read. Defaults to this checkout.

    Examples
    --------
    >>> inventory = discover()
    >>> len(inventory.proofs) >= 17
    True
    >>> any(
    ...     proof.pretty_name == "distance::kani_proofs::verify_cosine_bounded_3d"
    ...     for proof in inventory.proofs
    ... )
    True

    Returns
    -------
    Inventory
        The attributable declarations and the constructs that could not be
        attributed.
    """
    root = Path(root).resolve()
    proofs: list[ProofDeclaration] = []
    unresolved: list[UnresolvedDeclaration] = []
    by_package: dict[str, list[_Package]] = {}
    for package in _packages(root):
        by_package.setdefault(package.name, []).append(package)
    for name, roots in by_package.items():
        reader = _Reader(name, root)
        for package in roots:
            reader.visit(
                package.root,
                _Context((), (), package.is_binary),
                is_crate_root=True,
            )
        proofs.extend(reader.proofs)
        unresolved.extend(reader.unresolved)
        for path in _unreached(roots, reader.visited):
            unresolved.append(
                UnresolvedDeclaration(
                    kind="unreachable",
                    path=path.relative_to(root).as_posix(),
                    line=1,
                    detail=(
                        "proof-bearing source is not reached by any module "
                        "declaration from the crate root"
                    ),
                )
            )
    proofs.sort(key=lambda proof: (proof.package, proof.pretty_name))
    unresolved.sort(key=lambda report: (report.path, report.line, report.kind))
    return Inventory(proofs=tuple(proofs), unresolved=tuple(unresolved))