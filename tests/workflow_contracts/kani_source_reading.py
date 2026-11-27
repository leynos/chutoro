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

from rust_source_reading import (
    _FUNCTION,
    _blank,
    _brace_depths,
    _matching_brace,
    _regions,
)

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
#:
#: The match starts at the declaration's first word rather than at the ``mod``
#: keyword, because both a visibility qualifier and an attribute may sit in
#: front of the keyword: ``#[path = "alternate.rs"] pub mod first;`` puts
#: ``pub`` between them, and a walk that starts at ``mod`` reads the qualifier
#: as an unrelated token and scans back across the previous declaration's
#: semicolon to an attribute that belongs to that earlier item.
#:
#: A word boundary keeps ``model`` and ``modular`` out. Comments and literals
#: are blanked before this runs, so a mention in either has already gone.
_MOD_ITEM = re.compile(
    r"(?<![A-Za-z0-9_])(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<terminator>[;{])"
)

#: The ``#[path = "..."]`` attribute on a module declaration. Rust accepts an
#: ordinary string or a raw string for the value, and both name the file; any
#: other shape is matched by :data:`_ANY_PATH_ATTRIBUTE` and reported rather
#: than read as an absent attribute, because an unread value would send the
#: reader to the default location and hide a proof that moved.
_PATH_ATTRIBUTE = re.compile(
    r'#\[\s*path\s*=\s*(?:'
    r'"(?P<value>[^"]*)"'
    r'|r(?P<hashes>#*)"(?P<raw>.*?)"(?P=hashes)'
    r')\s*\]',
    re.DOTALL,
)

#: A ``path`` assignment inside another attribute, as ``cfg_attr`` writes it.
#: The value a conditional path names depends on configuration this reader does
#: not evaluate, so a declaration carrying one is reported rather than resolved.
_CONDITIONAL_PATH = re.compile(r"\bpath\s*=")

#: A macro definition whose body might contain proof declarations. Kani cannot
#: see a harness that only exists after expansion in a way this reader can
#: attribute to a module path, so the contract reports one rather than guessing.
#:
#: The body may be delimited by braces, parentheses or brackets; all three are
#: ordinary, and recognising only braces would miss a proof inside the others.
_MACRO_DEFINITION = re.compile(
    r"\b(?:macro_rules\s*!\s*[A-Za-z_][A-Za-z0-9_]*|macro\s+[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?P<opening>[{([])"
)

#: The closing delimiter for each macro-body opening.
_MACRO_CLOSING: typ.Final[dict[str, str]] = {"{": "}", "(": ")", "[": "]"}

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
    path_unreadable
        Whether the declaration carries a ``#[path]`` whose value this reader
        cannot read. Such a declaration is reported rather than resolved by
        the default rule, because guessing would look for the module in a
        place the attribute does not name.
    attributes
        The attributes attached to the declaration, each as its own text. A
        ``#[cfg(...)]`` here gates every declaration inside an inline module,
        which is how a test-only proof can sit behind a module.
    """

    name: str
    start: int
    end: int
    inline: bool
    path_value: str | None
    path_unreadable: bool = False
    attributes: tuple[str, ...] = ()


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
    inner
        The file's inner attributes (``#![...]``), which gate every
        declaration in it.
    """

    text: str
    code: str
    declarations: str
    modules: tuple[_ModuleItem, ...]
    macros: tuple[tuple[int, int], ...]
    inner: tuple[str, ...] = ()


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


def _attribute_start(text: str, end: int) -> int | None:
    """Return the offset of the ``#[`` opening the attribute ending at `end`.

    The closing bracket is matched by counting backwards, so an attribute whose
    body contains brackets -- a ``cfg_attr``, or a path value with one -- is
    read whole rather than truncated at its first ``]``.
    """
    depth = 0
    for offset in range(end - 1, -1, -1):
        character = text[offset]
        if character == "]":
            depth += 1
            continue
        if character != "[":
            continue
        depth -= 1
        # The depth reaching zero at a ``[`` is the bracket that opened the
        # attribute; the ``#`` immediately before it is the attribute's sigil.
        # Splitting the test keeps each branch single-purpose, and the depth
        # check comes first because an inner bracket closes while depth is
        # still positive.
        if depth != 0:
            continue
        if offset > 0 and text[offset - 1] == "#":
            return offset - 1
    return None


def _attached_attributes(
    declarations: str, code: str, start: int
) -> list[tuple[int, int]]:
    """Return the extents of the attributes attached at `start`.

    `declarations` keeps the ``#[path]`` value and decides what text an
    attribute carries; `code` has literals blanked and decides where its
    brackets are. A ``[`` or ``]`` inside a doc string is text, and counting it
    would put the attribute's opening at the wrong offset.

    Attributes are found by working backwards from the declaration, so an
    attribute on the same line and one spread over several lines are both read.
    A line-based walk fails both: it sees ``#[path = "x.rs"] mod first;`` as
    code because the line does not start with ``#[``, and it stops at the first
    continuation line of a multi-line attribute.

    The walk stops at the first non-whitespace that is not the end of an
    attribute, which is the previous item's own code. Reading further would let
    a `#[path]` on an earlier item be attached to this one, and a valid module
    tree would fail the manifest contract.

    A brace ends the walk too, because it separates items rather than
    decorating one. Without that stop, the closing brace of an enclosing inline
    module would be skipped and that module's `#[path]` adopted by its first
    child -- which is how ``#[path = "dir"] mod outer { #[path = "x.rs"] mod
    inner; }`` would give ``inner`` the directory meant for ``outer``.

    The walk also requires the character before the declaration to be an
    attribute's closing ``]``. A declaration carries its visibility qualifier
    in `start`, so anything else here is the previous item's own last token,
    and scanning back from it would attach that item's attribute to this one.

    Each attribute's extent is returned rather than its text, so the caller
    can read the value from `declarations` and the structure from `code`: an
    attribute's value may itself contain text shaped like another attribute --
    a ``#[doc]`` string naming ``#[path = "x.rs"]`` is the ordinary way to
    write documentation about one -- and a search across joined text would read
    that mention as the declaration's own path.
    """
    attached: list[tuple[int, int]] = []
    position = start
    while True:
        while position > 0 and code[position - 1].isspace():
            position -= 1
        if position == 0 or code[position - 1] in "{};":
            break
        if code[position - 1] != "]":
            break
        end = position
        opening = _attribute_start(code, end)
        if opening is None:
            break
        attached.insert(0, (opening, end))
        position = opening
    return attached


def _proof_attributes(code: str, start: int) -> list[str]:
    """Return the attributes above the declaration at `start`, innermost first.

    Comments and literals are already blanked in `code`, so a mention in
    either is not an attribute. The walk works backwards by offset rather than
    by line, because Rust does not require an attribute to occupy a line of its
    own: ``mod proofs { #[kani::proof]`` puts one after the module's opening
    brace, and a line-based walk stops at the ``mod`` and loses the proof.

    The walk stops at the first non-whitespace that is not an attribute's
    closing ``]``, and then at that attribute's ``#``. A brace ends it too:
    a brace separates items rather than decorating one, so the enclosing
    module's own attribute cannot be adopted by its first child.
    """
    found: list[str] = []
    position = start
    while True:
        while position > 0 and code[position - 1].isspace():
            position -= 1
        if position == 0 or code[position - 1] in "{};":
            break
        end = position
        opening = _attribute_start(code, end)
        if opening is None:
            break
        found.insert(0, _normalised(code[opening:end]))
        position = opening
    return found


def _attribute_end(text: str, opening: int) -> int | None:
    """Return one past the bracket closing the ``[`` at `opening`."""
    depth = 0
    for offset in range(opening, len(text)):
        depth += _bracket_delta(text[offset])
        if depth == 0 and text[offset] == "]":
            return offset + 1
    return None


def _bracket_delta(character: str) -> int:
    """Return how `character` changes the attribute's bracket depth."""
    if character == "[":
        return 1
    if character == "]":
        return -1
    return 0


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


def _macro_spans(code: str) -> tuple[tuple[int, int], ...]:
    """Return the extents of macro definition bodies in `code`."""
    spans: list[tuple[int, int]] = []
    for match in _MACRO_DEFINITION.finditer(code):
        opening = match.end() - 1
        closing = _matching_delimiter(code, opening, match["opening"])
        if closing is not None:
            spans.append((opening, closing))
    return tuple(spans)


def _matching_delimiter(code: str, opening: int, delimiter: str) -> int | None:
    """Return the offset of the delimiter closing the one at `opening`.

    A macro body may be delimited by any of the three bracket pairs, and only
    the same pair closes it: a ``}`` inside a parenthesised body is part of the
    body's tokens, not its end. Nesting of the same pair is counted, which is
    the shape a macro body takes.
    """
    closing = _MACRO_CLOSING[delimiter]
    depth = 0
    for offset in range(opening, len(code)):
        depth += _delimiter_delta(code[offset], delimiter, closing)
        if depth == 0:
            return offset
    return None


def _delimiter_delta(character: str, opening: str, closing: str) -> int:
    """Return how `character` changes the delimiter depth."""
    if character == opening:
        return 1
    if character == closing:
        return -1
    return 0


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


class _Lexed(typ.NamedTuple):
    """One source's code, its comment-blanked twin, and its scanner state.

    Grouped so the module reader takes one argument rather than four, and so
    the three texts travel together for the reasons recorded on each.
    """

    code: str
    declarations: str
    depths: list[int]
    macros: tuple[tuple[int, int], ...]
    inner: tuple[str, ...]


def _lex(text: str) -> _Lexed:
    """Return the scanner state for one source's text.

    Two texts are kept. `code` has comments and literals blanked, so an
    attribute inside either is not read. `declarations` has only comments
    blanked, which keeps the ``#[path]`` value: it is a string literal, and the
    fully blanked text has erased exactly what the reader needs. Positions are
    preserved in both, so every offset is the source's own.
    """
    regions = _regions(text)
    code = _blank(text, regions)
    declarations = _blank(
        text, (region for region in regions if region.is_comment)
    )
    depths = _brace_depths(code)
    return _Lexed(
        code=code,
        declarations=declarations,
        depths=depths,
        macros=_macro_spans(code),
        inner=_inner_attributes(code),
    )


def _module_item(lexed: _Lexed, match: re.Match[str]) -> _ModuleItem | None:
    """Return one module declaration, or None when it sits in a macro body."""
    if _inside(match.start(), lexed.macros):
        return None
    inline = match["terminator"] == "{"
    extents = _attached_attributes(lexed.declarations, lexed.code, match.start())
    pairs = [
        (lexed.declarations[start:end], lexed.code[start:end])
        for start, end in extents
    ]
    value, unreadable = _declared_path(pairs)
    attributes = tuple(text for text, _ in pairs)
    if inline:
        # An inline module's own inner attributes (``mod p { #![cfg(test)] }``)
        # gate everything inside it, exactly as a file's leading ones do.
        attributes += _inner_attributes(lexed.code[match.end() :])
    return _ModuleItem(
        name=match["name"],
        start=match.start(),
        end=_module_end(lexed, match) if inline else match.end(),
        inline=inline,
        path_value=value,
        path_unreadable=unreadable,
        attributes=attributes,
    )


def _declared_path(
    attributes: typ.Iterable[tuple[str, str]],
) -> tuple[str | None, bool]:
    """Return a declaration's ``#[path]`` value and whether it is unreadable.

    Each attribute is a pair: the text that carries its value, and the same
    span with literals blanked. Only an attribute whose own path is ``path``
    is read, so a ``#[doc]`` string that mentions one is not mistaken for a
    real assignment, and a conditional path is searched in the blanked text so
    an example inside a doc string is not read as an assignment. A path in a
    shape the reader cannot read is reported as unreadable rather than treated
    as absent, because the default rule would look for the module where the
    attribute does not point.
    """
    for text, structure in attributes:
        found = _PATH_ATTRIBUTE.match(text)
        if found is not None:
            return _path_value(found), False
        path = _attribute_path(text)
        if path == "path":
            return None, True
        if path == "cfg_attr" and _CONDITIONAL_PATH.search(structure):
            return None, True
    return None, False


def _path_value(found: re.Match[str]) -> str:
    """Return the path an attribute match carries, quoted or raw."""
    value = found["value"]
    return value if value is not None else found["raw"]


#: A ``cfg`` guard and its predicate body.
_CFG_GUARD = re.compile(r"#!?\[\s*cfg\s*\((?P<body>.*)\)\s*\]$", re.DOTALL)

#: One predicate: a name, or a combinator over parenthesised predicates.
_COMBINATOR = re.compile(r"(?P<fn>all|any|not)\s*\((?P<args>.*)\)$", re.DOTALL)


def _gated_by_test(attributes: typ.Iterable[str]) -> bool:
    """Return whether these guards exclude the item when the test cfg is off.

    The gate compiles without ``cfg(test)``, so a guard that evaluates false
    under that assumption means the item is not in the gate's build. The
    predicate is evaluated rather than searched for the word ``test``:
    ``cfg(not(test))`` and ``cfg(any(kani, test))`` both compile, and
    ``cfg(all(not(feature = "f"), test))`` does not.
    """
    return any(_excluded_without_test(attribute) for attribute in attributes)


def _excluded_without_test(attribute: str) -> bool:
    """Return whether one ``cfg`` attribute excludes the item under the gate."""
    found = _CFG_GUARD.match(attribute.strip())
    if found is None:
        return False
    return _predicate(found["body"]) is False


def _predicate(body: str) -> bool | None:
    """Evaluate one cfg predicate with ``test`` off and ``kani`` on.

    Unknown configuration -- a feature, a target, an OS -- is unknown rather
    than assumed, so only a predicate that is definitely false excludes an
    item from the gate's build.
    """
    body = body.strip()
    if body == "test":
        return False
    if body == "kani":
        return True
    found = _COMBINATOR.match(body)
    if found is None:
        return None
    values = [_predicate(argument) for argument in _arguments(found["args"])]
    return _combine(found["fn"], values)


def _arguments(body: str) -> list[str]:
    """Split one combinator's body at its top-level commas."""
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(body):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(body[start:index])
            start = index + 1
    parts.append(body[start:])
    return [part for part in (part.strip() for part in parts) if part]


def _combine(name: str, values: list[bool | None]) -> bool | None:
    """Combine sub-predicates under one combinator's three-valued logic."""
    if name == "not":
        return _negate(values[0] if values else None)
    if name == "all":
        if False in values:
            return False
        return None if None in values else True
    if True in values:
        return True
    return None if None in values else False


def _negate(value: bool | None) -> bool | None:
    """Return the three-valued negation of `value`."""
    return None if value is None else not value


def _inner_attributes(code: str) -> tuple[str, ...]:
    """Return a file's leading inner attributes, in source order.

    An inner attribute is written ``#![cfg(test)]`` and applies to the whole
    module, so the first declaration in the file inherits it.
    """
    found: list[str] = []
    position = 0
    while True:
        while position < len(code) and code[position].isspace():
            position += 1
        if not code.startswith("#![", position):
            return tuple(found)
        opening = position + 1
        closing = _attribute_end(code, opening)
        if closing is None:
            return tuple(found)
        found.append(_normalised(code[position:closing]))
        position = closing


def _module_end(lexed: _Lexed, match: re.Match[str]) -> int:
    """Return one past the closing brace of an inline module."""
    closing = _matching_brace(lexed.code, lexed.depths, match.end() - 1)
    return closing + 1 if closing is not None else len(lexed.code)


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
    lexed = _lex(text)
    items = [
        item
        for match in _MOD_ITEM.finditer(lexed.code)
        if (item := _module_item(lexed, match)) is not None
    ]
    return _Source(
        text=text,
        code=lexed.code,
        declarations=lexed.declarations,
        modules=tuple(items),
        macros=lexed.macros,
        inner=lexed.inner,
    )
