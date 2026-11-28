"""The records a Kani proof inventory holds.

The walker in :mod:`kani_harness_reading` produces these records; the
workflow contracts compare them with ``tools/kani/harnesses.toml``. Keeping
them here, apart from the walk, lets a test name the shape it expects without
importing the scanner.

Re-use policy: import from :mod:`kani_harness_reading`,
:mod:`kani_reader_support` and the reader fixtures only.
"""

from __future__ import annotations

import dataclasses
import typing as typ
from pathlib import Path


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
