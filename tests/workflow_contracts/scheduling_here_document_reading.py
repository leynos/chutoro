"""Read a shell here-document for where its body begins and ends.

A here-document's body is standard input for the command that opened it, not a
command the shell runs. Read as commands, a body that named the installer
looked like an installation, so ``cat <<'EOF'`` alone satisfied an ordering
contract that ran nothing. This module reads the opening line for the words
that close its documents and reads body lines against them.

The shell's body rules differ with the delimiter's spelling. A delimiter
quoted in any part — ``<<'EOF'``, ``<<"EOF"`` or ``<<E"O"F`` — makes the body
literal: nothing is expanded, and a backslash-newline pair is kept as the two
characters it was written as. An unquoted delimiter is expanded, and its
backslash-newline pairs are deleted before the closing line is compared, so a
body line ending in an odd number of backslashes joins the next one. ``<<-``
strips leading tabs from the body and its closing line; ``<<`` requires the
closing line to repeat the word exactly.

A ``<<`` in an arithmetic expansion is read as an opening too, which can only
make later lines look like body text, and a body the reader never closes
consumes the rest of the script: the reading then finds no evidence rather
than false evidence, so the mistake fails the contract instead of passing it.

Re-use policy: import from ``scheduling_line_reading`` only, the one reader
that drops a document's body. A new here-document rule belongs here with the
shell behaviour it models stated beside it.
"""

from __future__ import annotations

import typing as typ

from scheduling_quote_reading import mask_quotes, restore_quotes

#: Characters that end a here-document's delimiter word. The shell lets a
#: metacharacter follow it, as ``cat <<EOF; make test`` does, so the word ends
#: where the operator begins.
_DELIMITER_ENDS = frozenset(" \t;&|()<>")


class HereDocument(typ.NamedTuple):
    """One here-document a command line opened."""

    word: str
    strips_tabs: bool
    literal: bool


def _delimiter_start(masked: str, start: int) -> int:
    """Return the index where a ``<<`` operator's delimiter word begins.

    ``<<-`` strips leading tabs from the body, and either form may put spaces
    between the operator and the word.

    Examples
    --------
    >>> _delimiter_start("<<EOF", 2)
    2
    >>> _delimiter_start("<<-EOF", 2)
    3
    >>> _delimiter_start("<< EOF", 2)
    3
    """
    if masked.startswith("-", start):
        start += 1
    while start < len(masked) and masked[start] in " \t":
        start += 1
    return start


def _delimiter_end(masked: str, start: int) -> int:
    """Return the index where the delimiter word beginning at `start` ends.

    Examples
    --------
    >>> _delimiter_end("EOF; make test", 0)
    3
    >>> _delimiter_end("EOF", 0)
    3
    """
    end = start
    while end < len(masked) and masked[end] not in _DELIMITER_ENDS:
        end += 1
    return end


def _unescaped(masked: str) -> str:
    """Return `masked` with the backslash of each unquoted escape removed.

    ``mask_quotes`` keeps the backslash in front of the marker it puts in
    place of an escaped character, because the word it restores may need it.
    The shell removes that backslash when it forms the word, so the delimiter
    of ``<<\\EOF`` is ``EOF`` and the closing line is ``EOF``, not ``\\EOF``.
    A backslash inside quoted text is part of that text and stays.

    Examples
    --------
    >>> _unescaped("EOF")
    'EOF'
    >>> escaped = chr(92) + chr(1) + "0" + chr(2) + "b"
    >>> _unescaped(escaped) == chr(1) + "0" + chr(2) + "b"
    True
    """
    kept: list[str] = []
    index = 0
    while index < len(masked):
        if masked[index] == "\\" and masked[index + 1 : index + 2] == "\x01":
            index += 1
            continue
        kept.append(masked[index])
        index += 1
    return "".join(kept)


def _is_literal(raw: str, word: str) -> bool:
    """Return whether a delimiter's spelling quotes any part of it.

    Quoted text is set aside before the line is read, so a quoted delimiter
    differs from its restored form. Any quoting at all makes the body literal.

    Examples
    --------
    >>> _is_literal("EOF", "EOF")
    False
    >>> _is_literal("E" + chr(1) + "0" + chr(2), "EOF")
    True
    """
    return raw != word


def here_documents(line: str) -> list[HereDocument]:
    """Return the here-documents `line` opens, outermost first.

    Quoted text is data, so a ``<<`` inside quotes opens nothing; the
    delimiter itself may be quoted, and the quoted form is restored because
    that is the word the closing line repeats while the body stays literal. A
    backslash escapes a character of the delimiter just as it escapes one of
    any word, so ``<<\\EOF`` waits for ``EOF``.

    Examples
    --------
    >>> here_documents("cat <<'EOF'")
    [HereDocument(word='EOF', strips_tabs=False, literal=True)]
    >>> here_documents("cat <<-EOF")
    [HereDocument(word='EOF', strips_tabs=True, literal=False)]
    >>> here_documents("echo '<<EOF'")
    []
    >>> here_documents("cat <<EOF; make test")
    [HereDocument(word='EOF', strips_tabs=False, literal=False)]
    >>> documents = here_documents("cat <<A <<B")
    >>> [document.word for document in documents]
    ['A', 'B']
    >>> escaped = here_documents("cat <<" + chr(92) + "EOF")[0]
    >>> escaped.word, escaped.literal
    ('EOF', True)
    """
    masked = mask_quotes(line)
    documents: list[HereDocument] = []
    index = masked.text.find("<<")
    while index != -1:
        strips_tabs = masked.text.startswith("-", index + 2)
        start = _delimiter_start(masked.text, index + 2)
        end = _delimiter_end(masked.text, start)
        if end > start:
            raw = masked.text[start:end]
            word = restore_quotes(_unescaped(raw), masked.quoted)
            literal = _is_literal(raw, word)
            documents.append(HereDocument(word, strips_tabs, literal))
        index = masked.text.find("<<", max(end, index + 2))
    return documents


def ends_document(document: HereDocument, line: str) -> bool:
    """Return whether `line` is the closing line of `document`.

    The closing line repeats the delimiter exactly for ``<<``, so a leading
    space is body text and ``EOF `` is not the word ``EOF``. A ``<<-``
    document ignores leading tabs on both its body and its closing line.

    Examples
    --------
    >>> exact = HereDocument("EOF", False, False)
    >>> ends_document(exact, " EOF")
    False
    >>> ends_document(exact, "EOF ")
    False
    >>> ends_document(exact, "EOF")
    True
    >>> stripping = HereDocument("EOF", True, False)
    >>> ends_document(stripping, chr(9) + "EOF")
    True
    >>> ends_document(stripping, " EOF")
    False
    """
    candidate = line.lstrip("\t") if document.strips_tabs else line
    return candidate == document.word