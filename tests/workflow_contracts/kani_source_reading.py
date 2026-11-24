"""Parse one Rust source for Kani proofs and its module declarations.

The module-level reader divides into two halves. This half answers questions
about a single file: which attributes above a function make it a proof, which
``mod`` declarations it holds, where a macro definition begins and ends, and
which inline modules enclose an offset. The other half,
:mod:`kani_harness_reading`, walks the crate tree and attributes declarations
to a package.

The division is the 400-line limit's: the two halves together are one
scanner, and splitting them is cheaper than a second scanner that would
eventually disagree with this one.

Re-use policy: import from :mod:`kani_harness_reading` only.
"""

from __future__ import annotations

import dataclasses
import re
import typing as typ
from pathlib import Path

from rust_declaration_reading import _brace_depths, _matching_brace
from rust_source_reading import _FUNCTION, _attributes_before, _blank, _regions

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


def _attached_attributes(declarations: str, start: int) -> str:
    """Return the attributes attached to the item declared at `start`.

    `_attributes_before` returns every preceding line, so joining all of it
    would let a `#[path]` on an earlier item be read as this one's: after
    `#[path = "alternate.rs"] mod first;`, an ordinary `mod second;` would
    inherit `alternate.rs`, its real file would be reported unreachable, and a
    valid module tree would fail the manifest contract.

    The walk therefore stops at the first preceding line that is not an
    attribute or a blank line, which is the previous item's own code. Literals
    are preserved in `declarations`, so the `#[path]` value survives.
    """
    attached: list[str] = []
    for line in reversed(declarations[:start].splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#["):
            break
        attached.insert(0, stripped)
    return " ".join(attached)


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
        found = _PATH_ATTRIBUTE.search(_attached_attributes(declarations, match.start()))
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


