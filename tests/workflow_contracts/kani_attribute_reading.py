"""Read the Rust attributes that turn a function into a Kani proof.

An attribute is not a line: it may share a line with the declaration it
qualifies, and it may span several lines. It may also carry brackets in its
body, as ``cfg_attr`` and a path value can, and a bracket inside a doc string
is text rather than structure. This module reads attributes by offset in text
whose literals have been blanked, which is what makes all of those shapes come
out right.

The functions here answer only what an attribute means: its path, its extent,
whether a proof attribute attaches at a given declaration, and what kind of
proof registration it represents. Reading a file's module declarations belongs
to :mod:`kani_source_reading`.

Re-use policy: import from :mod:`kani_source_reading` and
:mod:`kani_harness_reading` only. Attribute syntax is Rust's, so a second
consumer should extend this module rather than restate the reading.
"""

from __future__ import annotations

import re
import typing as typ

#: The attribute paths that make a function a Kani harness. ``proof`` is the
#: ordinary bounded harness; ``proof_for_contract`` is the function-contract
#: entry point. Both are declarations the inventory must classify.
PROOF_ATTRIBUTES = frozenset({"kani::proof", "kani::proof_for_contract"})

#: One attribute's path, with Rust's optional spacing around ``::`` removed by
#: the caller before matching.
_ATTRIBUTE_PATH = re.compile(r"^#\[\s*(?P<path>[A-Za-z_][A-Za-z0-9_:]*)")

#: Rust permits whitespace around ``::``. Attributes are normalised through
#: this before their path is read, so ``#[kani :: proof]`` is one attribute.
_SPACED_PATH = re.compile(r"\s*(::)\s*")


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


