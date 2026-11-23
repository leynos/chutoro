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
import tomllib
import typing as typ
from pathlib import Path

from rust_source_reading import _FUNCTION

from kani_source_reading import (
    _ModuleItem,
    _Source,
    _inside,
    _line_of,
    _module_chain,
    _parse,
    _proof_kind,
    _proof_attributes,
)
from workflow_support import ROOT

#: Directories the sweep never enters. They hold build output, caches, or
#: another repository's checkout rather than this workspace's sources.
IGNORED_DIRECTORIES = frozenset({
    ".git",
    ".uv-cache",
    ".uv-tools",
    ".verus",
    "node_modules",
    "target",
})

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
    """

    name: str
    root: Path
    directory: Path


def _child_directory(file: Path, is_crate_root: bool) -> Path:
    """Return the directory in which `file` resolves its child modules."""
    if is_crate_root or file.name == "mod.rs":
        return file.parent
    return file.with_suffix("")


def _path_base(
    file: Path,
    is_crate_root: bool,
    chain: tuple[_ModuleItem, ...],
) -> Path:
    """Return the directory a ``#[path]`` value is relative to.

    Rust resolves a path attribute on a declaration outside an inline module
    block relative to the containing file's directory, and one inside an
    inline block relative to the module directory with the inline blocks as
    directories. A "mod-rs" file — a crate root or ``mod.rs`` — starts from its
    own directory; any other file starts from its stem directory.
    """
    if not chain:
        return file.parent
    if is_crate_root or file.name == "mod.rs":
        base = file.parent
    else:
        base = file.with_suffix("")
    return base.joinpath(*(item.name for item in chain))


def _child_file(
    file: Path,
    is_crate_root: bool,
    chain: tuple[_ModuleItem, ...],
    name: str,
    path_value: str | None,
) -> Path | None:
    """Return the file a ``mod name;`` declaration reads, if one exists."""
    if path_value is not None:
        candidate = _path_base(file, is_crate_root, chain) / path_value
        return candidate if candidate.is_file() else None
    directory = _child_directory(file, is_crate_root).joinpath(
        *(item.name for item in chain)
    )
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
        self._visited: set[Path] = set()
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
        module: tuple[str, ...],
        *,
        is_crate_root: bool = False,
    ) -> None:
        """Read `file` as the module `module`, following its declarations."""
        resolved = file.resolve()
        if resolved in self._visited:
            return
        self._visited.add(resolved)
        source = _parse(file)
        if source is None:
            return
        relative = file.relative_to(self._repo_root).as_posix()
        self._record_proofs(source, module, relative)
        self._follow(file, source, module, is_crate_root)

    def _record_proofs(
        self,
        source: _Source,
        module: tuple[str, ...],
        relative: str,
    ) -> None:
        """Record each proof in one file, or report why it cannot be."""
        for match in _FUNCTION.finditer(source.code):
            kind = _proof_kind(_proof_attributes(source.code, match.start()))
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
            pretty = "::".join(
                (*module, *(item.name for item in chain), match["name"])
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
        self,
        file: Path,
        source: _Source,
        module: tuple[str, ...],
        is_crate_root: bool,
    ) -> None:
        """Follow each file-based module declaration in `file`."""
        for item in source.modules:
            if item.inline:
                continue
            chain = _module_chain(source.modules, item.start)
            child = _child_file(
                file,
                is_crate_root,
                chain,
                item.name,
                item.path_value,
            )
            if child is None:
                if item.path_value is not None:
                    self._report(
                        "path-attribute",
                        file.relative_to(self._repo_root).as_posix(),
                        _line_of(source.text, item.start),
                        f"#[path = {item.path_value!r}] does not resolve to a "
                        "readable source file",
                    )
                continue
            self.visit(
                child,
                (*module, *(parent.name for parent in chain), item.name),
            )


def _package_directories(root: Path) -> list[tuple[str, Path]]:
    """Return each package's name and directory under `root`."""
    found: list[tuple[str, Path]] = []
    for manifest in sorted(root.rglob("Cargo.toml")):
        if IGNORED_DIRECTORIES.intersection(manifest.relative_to(root).parts):
            continue
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
        package = document.get("package")
        if isinstance(package, dict) and (manifest.parent / "src").is_dir():
            found.append((package["name"], manifest.parent))
    return found


def _packages(root: Path) -> list[_Package]:
    """Return each package that has a library or binary crate root."""
    packages: list[_Package] = []
    for name, directory in _package_directories(root):
        for candidate in ("lib.rs", "main.rs"):
            crate_root = directory / "src" / candidate
            if crate_root.is_file():
                packages.append(
                    _Package(name=name, root=crate_root, directory=directory)
                )
                break
    return packages


def _holds_proof(source: _Source) -> bool:
    """Return whether `source` declares at least one proof-shaped function."""
    return any(
        _proof_kind(_proof_attributes(source.code, match.start())) is not None
        for match in _FUNCTION.finditer(source.code)
    )


def _unreached(package: _Package, visited: frozenset[Path]) -> list[Path]:
    """Return proof-bearing sources under `package` that no module reached."""
    found: list[Path] = []
    for path in sorted((package.directory / "src").rglob("*.rs")):
        if path.resolve() in visited:
            continue
        source = _parse(path)
        if source is not None and _holds_proof(source):
            found.append(path)
    return found


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
    proofs: list[ProofDeclaration] = []
    unresolved: list[UnresolvedDeclaration] = []
    for package in _packages(root):
        reader = _Reader(package.name, root)
        reader.visit(package.root, (), is_crate_root=True)
        proofs.extend(reader.proofs)
        unresolved.extend(reader.unresolved)
        for path in _unreached(package, reader.visited):
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