"""Read a Make declaration for the targets it names.

The scheduling contract checks that its target is declared ``.PHONY``, because
a same-named file would otherwise let ``make`` report the target as up to date
and skip the gate. Reading that declaration is Make syntax rather than shell
syntax: a comment starts at an unescaped ``#`` anywhere on the line, a
backslash continues a declaration onto the next line, and a comment continued
by a backslash, a ``define`` body or a conditional block declares nothing.

This module holds the phony-target reading and the rule-header reading. The
directive structure it relies on lives in ``makefile_structure_reading``, and
the split keeps each file inside the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_contract_test`` and
``deny_gate_test``, the two places that assert over it. A new Make rule for a
declaration belongs here with the make behaviour it models stated beside it.
"""

from __future__ import annotations

from makefile_structure_reading import (
    declaring_lines,
    declarations,
    make_comment_start,
)

#: The characters a backslash quotes in a Make target name.
_MAKE_QUOTED = frozenset(" \t#:;=\\\\")


def _target_names(text: str) -> list[str]:
    """Return the target names `text` lists, with escapes resolved.

    A backslash escapes the character after it, so ``other\\ gate`` is one
    target name with a space in it rather than two names. An unescaped
    whitespace character separates names.

    Examples
    --------
    >>> _target_names("clean gate")
    ['clean', 'gate']
    >>> _target_names("other" + chr(92) + " gate")
    ['other gate']
    >>> _target_names("a" + chr(92) + "#1")
    ['a#1']
    """
    names: list[str] = []
    current = ""
    index = 0
    while index < len(text):
        char, index, escaped = _name_step(text, index)
        if escaped or char not in " \t":
            current += char
            continue
        if current:
            names.append(current)
            current = ""
    if current:
        names.append(current)
    return names


def _name_step(text: str, index: int) -> tuple[str, int, bool]:
    """Return the character at `index`, the index after it and its escape.

    A backslash quotes only the characters Make reads specially — whitespace,
    ``#``, ``:``, ``;``, ``=`` and another backslash — so the pair contributes
    that character alone and an escaped space belongs to the name rather than
    separating names. A backslash before an ordinary character is part of the
    name.

    Examples
    --------
    >>> _name_step("a", 0)
    ('a', 1, False)
    >>> _name_step(chr(92) + " ", 0)
    (' ', 2, True)
    >>> _name_step(chr(92) + "g", 0)
    ('\\\\', 1, False)
    """
    if text[index] == "\\" and text[index + 1 : index + 2] in _MAKE_QUOTED:
        return text[index + 1], index + 2, True
    return text[index], index + 1, False


def _declared_names(line: str) -> list[str]:
    """Return the target names one declaration line adds.

    The continuation backslash ends the line rather than naming a target, an
    escaped ``#`` names a target that contains one, and anything from an
    unescaped ``#`` on is a Make comment rather than a name.

    Examples
    --------
    >>> _declared_names(" clean " + chr(92))
    ['clean']
    >>> _declared_names(" clean" + chr(92) + "#1")
    ['clean#1']
    >>> _declared_names(" clean # tidy up")
    ['clean']
    >>> _declared_names(" other" + chr(92) + " gate")
    ['other gate']
    """
    comment = make_comment_start(line)
    uncommented = line if comment is None else line[:comment]
    return _target_names(uncommented.rstrip().removesuffix("\\"))


def _declared_targets(declaration: str) -> list[str]:
    """Return the names one ``.PHONY`` declaration adds, if it declares any.

    A declaration carrying an ``=`` is an assignment rather than a phony
    prerequisite: ``.PHONY:= gate`` assigns the variable and
    ``.PHONY: gate = value`` assigns ``gate`` for that target, so neither
    names a target.

    Examples
    --------
    >>> _declared_targets(".PHONY: clean gate")
    ['clean', 'gate']
    >>> _declared_targets(".PHONY:= gate")
    []
    >>> _declared_targets(".PHONY: gate = value")
    []
    >>> _declared_targets(".PHONY : gate")
    ['gate']
    >>> _declared_targets(".PHONY: gate ; @echo status=ok")
    ['gate']
    >>> _declared_targets(chr(9) + ".PHONY: gate")
    []
    >>> _declared_targets("clean: gate")
    []
    """
    if declaration[:1] == "\t":
        return []
    header = declaration.split(";", 1)[0]
    if "=" in header:
        return []
    colon = header.find(":")
    if colon == -1 or _target_names(header[:colon]) != [".PHONY"]:
        return []
    return _declared_names(header[colon + 1 :])


def phony_targets(text: str) -> set[str]:
    """Return the names a ``.PHONY`` declaration covers.

    Make joins a continued declaration into one line before it reads the
    comment on it, so the declaration is collapsed first and the comment is
    then removed: ``.PHONY: clean # gate`` declares ``clean`` alone, and
    reading the comment would make a same-named file seem covered when the
    target could still be skipped. A comment ending in an unescaped backslash
    continues onto the next line, which declares nothing either, so
    ``.PHONY: clean # \\`` followed by ``gate`` still declares only ``clean``;
    a doubled backslash ends the comment, so the next line declares as usual.

    Examples
    --------
    >>> sorted(phony_targets(".PHONY: clean " + chr(35) + " gate"))
    ['clean']
    >>> phony_targets(".PHONY:= gate")
    set()
    >>> "gate" not in phony_targets(".PHONY: " + chr(92) + "gate")
    True
    >>> declaration = ".PHONY: clean " + chr(92) + chr(10) + " gate"
    >>> sorted(phony_targets(declaration))
    ['clean', 'gate']
    >>> continued = (
    ...     ".PHONY: clean " + chr(35) + " " + chr(92) + chr(10) + " gate"
    ... )
    >>> sorted(phony_targets(continued))
    ['clean']
    >>> commented = "# off " + chr(92) + chr(10) + ".PHONY: gate"
    >>> phony_targets(commented)
    set()
    >>> resumed = (
    ...     ".PHONY: clean " + chr(35) + " " + chr(92) + chr(10) + "gone"
    ...     + chr(10) + ".PHONY: gate"
    ... )
    >>> sorted(phony_targets(resumed))
    ['clean', 'gate']
    >>> doubled = (
    ...     ".PHONY: clean " + chr(35) + " " + chr(92) * 2 + chr(10)
    ...     + ".PHONY: gate"
    ... )
    >>> sorted(phony_targets(doubled))
    ['clean', 'gate']
    """
    names: set[str] = set()
    for declaration in declarations(declaring_lines(text)):
        comment = make_comment_start(declaration)
        if comment is not None:
            declaration = declaration[:comment]
        names.update(_declared_targets(declaration))
    return names

def declares_target(line: str, target: str) -> bool:
    """Return whether `line` is a rule header declaring `target`.

    GNU make allows whitespace between a target name and the colon, and one
    header may name several targets, so the names are read before the colon
    rather than matched as a prefix. A tab-indented line is recipe text, and
    ``:=`` is a variable assignment rather than a rule.

    Examples
    --------
    >>> declares_target("build: ; @:", "build")
    True
    >>> declares_target("build :", "build")
    True
    >>> declares_target("build test:", "build")
    True
    >>> declares_target("::;", "build")
    False
    >>> declares_target("build := 1", "build")
    False
    >>> declares_target("build ::= 1", "build")
    False
    >>> declares_target("build = prefix: suffix", "build")
    False
    >>> declares_target("build::", "build")
    True
    >>> declares_target(chr(9) + "build:", "build")
    False
    >>> declares_target("builder:", "build")
    False
    >>> declares_target("clean # build:", "build")
    False
    >>> escaped = "other" + chr(92) + " gate:"
    >>> declares_target(escaped, "gate")
    False
    >>> declares_target(escaped, "other gate")
    True
    """
    if line[:1] == "\t":
        return False
    comment = make_comment_start(line)
    header = line if comment is None else line[:comment]
    colon = header.find(":")
    if colon == -1 or "=" in header[:colon]:
        return False
    if header[colon:].lstrip(":").startswith("="):
        return False
    return target in _target_names(header[:colon])
