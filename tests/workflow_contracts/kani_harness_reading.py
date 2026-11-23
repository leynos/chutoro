"""Read every Kani proof declaration in the workspace tree.

The workflow-contract suite needs the whole proof surface, not merely the
files whose names happen to match ``kani_*``. Six proofs live in an inline
``#[cfg(kani)] mod kani_proofs`` block at the bottom of
``chutoro-core/src/distance/mod.rs``, and a filename-based inventory never saw
them. This module therefore reads declarations out of the source.

It reuses :mod:`rust_source_reading`'s lexer so a ``#[kani::proof]`` in a
comment or a string literal is not a declaration, and
:mod:`rust_declaration_reading`'s brace counting so the module path of a
declaration can be derived from the file it sits in plus the inline ``mod``
blocks that enclose it. A file-based module takes its path from the ``mod``
declaration chain from the crate root, which is what makes an unresolved
``#[path]`` visible instead of silently yielding a plausible-looking name.

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
import re
import tomllib
import typing as typ
from pathlib import Path

from rust_declaration_reading import _brace_depths, _matching_brace
from rust_source_reading import _FUNCTION, _attributes_before, _blank, _regions
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

#: The attribute paths that make a function a Kani harness. ``proof`` is the
#: ordinary bounded harness; ``proof_for_contract`` is the function-contract
#: entry point. Both are declarations the inventory must classify.
PROOF_ATTRIBUTES = frozenset({"kani::proof", "kani::proof_for_contract"})

#: One attribute's path, with Rust's optional spacing around ``::`` removed by
#: the caller before matching.
_ATTRIBUTE_PATH = re.compile(r"^#\[\s*(?P<path>[A-Za-z_][A-Za-z0-9_:]*)")

#: A module declaration. ``mod name;`` names a file; ``mod name {`` opens an
#: inline module. A visibility qualifier and indentation are both ordinary, and
#: ``pub(crate)`` is matched without treating its parenthesis as indentation.
_MOD_ITEM = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?mod[ \t]+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)[ \t]*(?P<terminator>[;{])",
    re.MULTILINE,
)

#: The ``#[path = "..."]`` attribute on a module declaration. Only a plain
#: string literal is resolvable; anything else is reported.
_PATH_ATTRIBUTE = re.compile(r'#\[\s*path\s*=\s*"(?P<value>[^"]*)"\s*\]')

#: A macro definition whose body might contain proof declarations. Kani cannot
#: see a harness that only exists after expansion in a way this reader can
#: attribute to a module path, so the contract reports one rather than guessing.
_MACRO_DEFINITION = re.compile(
    r"\b(?:macro_rules\s*!\s*[A-Za-z_][A-Za-z0-9_]*|macro\s+[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*\{"
)

#: Rust permits whitespace around ``::``. Attributes are normalised through
#: this before their path is read, so ``#[kani :: proof]`` is one attribute.
_SPACED_PATH = re.compile(r"\s*(::)\s*")


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


@dataclasses.dataclass(frozen=True)
class _ModuleItem:
    """One module declaration, inline or file-based.

    Attributes
    ----------
    name
        The identifier after ``mod``.
    start
        The offset of the declaration.
    end
        For an inline module, one past its closing brace; for a file module,
        one past the semicolon.
    inline
        Whether the declaration opens a body rather than naming a file.
    path_value
        The ``#[path = "..."]`` value, or None when the declaration carries
        none. A path attribute names the file that provides the module; it
        does not change the module's own name.
    """

    name: str
    start: int
    end: int
    inline: bool
    path_value: str | None


@dataclasses.dataclass(frozen=True)
class _Source:
    """One parsed source file.

    Attributes
    ----------
    text
        The file's contents.
    code
        The contents with comments and literals blanked. Positions are
        preserved, so every offset below is the source's own. Attribute paths
        are read from this text, because a ``#[kani::proof]`` inside a string
        literal is data rather than a declaration.
    declarations
        The contents with comments blanked and literals kept, which is where a
        ``#[path]`` value can still be read. A ``#[path]`` value is a string
        literal, so the fully blanked text has erased exactly what the reader
        needs.
    modules
        The module declarations in source order.
    macros
        The extents of macro definition bodies.
    """

    text: str
    code: str
    declarations: str
    modules: tuple[_ModuleItem, ...]
    macros: tuple[tuple[int, int], ...]


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


def _line_of(text: str, offset: int) -> int:
    """Return the one-based line containing `offset`."""
    return text.count("\n", 0, offset) + 1


def _normalised(attribute: str) -> str:
    """Return `attribute` with spacing around ``::`` removed."""
    return _SPACED_PATH.sub(r"\1", attribute.strip())


def _attribute_path(attribute: str) -> str | None:
    """Return an attribute's path, or None when the line is not an attribute."""
    found = _ATTRIBUTE_PATH.match(_normalised(attribute))
    return found["path"] if found is not None else None


def _proof_attributes(code: str, start: int) -> list[str]:
    """Return the attributes above the declaration at `start`, innermost first.

    Comments and literals are already blanked in `code`, so a mention in
    either is not an attribute. The walk stops at the first line that is not
    an attribute: that is the previous item's code, and nothing above it
    belongs to this declaration.
    """
    found: list[str] = []
    for line in _attributes_before(code[:start]):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#["):
            break
        found.append(_normalised(stripped))
    return found


def _proof_kind(attributes: typ.Iterable[str]) -> str | None:
    """Return what registers a function as a proof, or None.

    ``proof`` covers both accepted attribute forms, ``kani::proof`` and
    ``kani::proof_for_contract``. ``cfg-attr`` reports a proof whose existence
    depends on configuration.
    """
    for attribute in attributes:
        path = _attribute_path(attribute)
        if path in PROOF_ATTRIBUTES:
            return "proof"
        if path == "cfg_attr" and any(
            proof in attribute for proof in PROOF_ATTRIBUTES
        ):
            return "cfg-attr"
    return None


def _macro_spans(code: str, depths: list[int]) -> tuple[tuple[int, int], ...]:
    """Return the extents of macro definition bodies in `code`."""
    spans: list[tuple[int, int]] = []
    for match in _MACRO_DEFINITION.finditer(code):
        opening = match.end() - 1
        closing = _matching_brace(code, depths, opening)
        if closing is not None:
            spans.append((opening, closing))
    return tuple(spans)


def _inside(offset: int, spans: typ.Iterable[tuple[int, int]]) -> bool:
    """Return whether `offset` lies inside one of `spans`."""
    return any(start <= offset < end for start, end in spans)


def _module_chain(
    modules: typ.Iterable[_ModuleItem], offset: int
) -> tuple[_ModuleItem, ...]:
    """Return the inline modules enclosing `offset`, outermost first."""
    return tuple(
        module
        for module in modules
        if module.inline and module.start < offset < module.end
    )


def _parse(path: Path) -> _Source | None:
    """Return `path` parsed for module and proof structure, or None.

    A file that cannot be decoded returns None rather than raising, because
    the caller has already recorded that it exists and each caller decides
    whether an unreadable file is an offence. A proof-bearing file that is
    unreadable is reported by the sweep that expected to read it.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    regions = _regions(text)
    code = _blank(text, regions)
    # A `#[path]` value is a string literal, which the fully blanked text has
    # erased. Module structure is read from the comment-blanked text, where the
    # value survives; a comment holding a brace cannot move a module boundary
    # because a comment carries no attribute and no module declaration.
    declarations = _blank(text, (region for region in regions if region.is_comment))
    depths = _brace_depths(code)
    macros = _macro_spans(code, depths)
    items: list[_ModuleItem] = []
    for match in _MOD_ITEM.finditer(code):
        if _inside(match.start(), macros):
            continue
        inline = match["terminator"] == "{"
        if inline:
            closing = _matching_brace(code, depths, match.end() - 1)
            end = closing + 1 if closing is not None else len(code)
        else:
            end = match.end()
        attributes = " ".join(_attributes_before(declarations[: match.start()]))
        found = _PATH_ATTRIBUTE.search(attributes)
        items.append(
            _ModuleItem(
                name=match["name"],
                start=match.start(),
                end=end,
                inline=inline,
                path_value=found["value"] if found else None,
            )
        )
    return _Source(
        text=text,
        code=code,
        declarations=declarations,
        modules=tuple(items),
        macros=macros,
    )


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