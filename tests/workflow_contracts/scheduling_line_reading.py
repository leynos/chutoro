"""Join a CI step's physical lines into the command lines the shell reads.

A workflow step's script is one shell script, and the shell does not read it a
line at a time. A line ending in a backslash continues the current line; a
line ending in a list operator — ``&&``, ``||``, ``|`` or ``|&`` — continues
the current command list, and the next line becomes that operator's
right-hand side. Reading either as a line boundary splits one command into
two, which can hide a skipped installer (``true ||`` on one line, the
installer on the next) or invent a status-bearing last command that the shell
never runs separately.

This module performs both joins and drops a here-document's body, which is
standard input for the command that opened it; where that body ends is
``scheduling_here_document_reading``'s reading. ``scheduling_segment_reading``
reads the resulting lines as simple commands, and the split keeps every file
inside the 400-line limit ``AGENTS.md`` sets.

Re-use policy: import from ``scheduling_segment_reading`` only; this module
reuses the comment and quote scanning in ``scheduling_command_reading`` and
the here-document reading in ``scheduling_here_document_reading``. A new
line-level rule belongs here with the shell behaviour it models stated beside
it.
"""

from __future__ import annotations

from scheduling_comment_reading import strip_shell_comment
from scheduling_quote_reading import mask_quotes
from scheduling_here_document_reading import (
    HereDocument,
    ends_document,
    here_documents,
)

#: Operators that leave the command list open at the end of a line. Any of
#: them makes the next line the right-hand side of the list, so the two
#: physical lines are one command for the shell. A lone ``&`` backgrounds the
#: command and opens nothing, and ``;`` ends the list.
_LIST_OPERATORS = ("&&", "||", "|&", "|")


def _ends_with_operator(line: str) -> bool:
    """Return whether `line` ends with an unescaped list operator.

    Quoted text is data, so an operator inside quotes ends nothing, and a
    backslash makes the character after it literal, so an escaped ``&`` is
    part of a word rather than the start of one. Masking both before the
    check leaves only operators the shell would act on.

    Examples
    --------
    >>> _ends_with_operator("make test &&")
    True
    >>> _ends_with_operator("make test |")
    True
    >>> _ends_with_operator("echo '&&'")
    False
    >>> _ends_with_operator("make test " + chr(92) + "&&")
    False
    >>> _ends_with_operator("make test &")
    False
    """
    text = mask_quotes(line).text.rstrip()
    return any(text.endswith(operator) for operator in _LIST_OPERATORS)


def _escape_continuation(line: str) -> tuple[str, bool]:
    """Return `line` with a final escaped newline deleted, and that verdict.

    Here-document text is not parsed for quotes, so only a backslash that
    escapes the newline joins the next physical line.

    Examples
    --------
    >>> _escape_continuation("x " + chr(92))
    ('x ', True)
    >>> text, continues = _escape_continuation("x" + chr(92) * 2)
    >>> continues, len(text)
    (False, 3)
    """
    run = len(line) - len(line.rstrip("\\"))
    if run % 2 == 0:
        return line, False
    return f"{line[: len(line) - run]}{chr(92) * (run - 1)}", True


def _in_single_quotes(text: str) -> bool:
    """Return whether `text` ends inside an ordinary single-quoted string.

    The shell preserves every character of such a string, including a
    backslash-newline pair, so a line that ends inside one continues without
    the pair being deleted.

    Examples
    --------
    >>> _in_single_quotes("echo 'a")
    True
    >>> _in_single_quotes('echo "a')
    False
    >>> _in_single_quotes("echo 'a'")
    False
    """
    quote = ""
    index = 0
    while index < len(text):
        char, index = _quote_step(text, index, quote)
        if quote == "'" and char == "'":
            quote = ""
        elif not quote and char in "'\"":
            quote = char
    return quote == "'"


def _quote_step(text: str, index: int, quote: str) -> tuple[str, int]:
    """Return the character that decides quoting, and the index after it.

    A backslash escapes the character after it wherever it is read, so the
    pair is consumed whole and the escaped character is not returned as quote
    syntax.

    Examples
    --------
    >>> _quote_step('a"b', 0, "")
    ('a', 1)
    >>> escaped = chr(92) + '"'
    >>> _quote_step(escaped, 0, '"')
    ('', 2)
    """
    if quote != "'" and text[index : index + 1] == "\\":
        return "", index + 2
    return text[index : index + 1], index + 1


def _continuation(line: str) -> tuple[str, bool]:
    """Return what `line` leaves before its newline, and whether it continues.

    The shell deletes the final backslash and the newline with nothing
    inserted, so only an odd run continues and the rest is left as written for
    the later reading to pair up. Three backslashes therefore leave two, which
    the lexer reads as one literal backslash followed by an operator, rather
    than one, which it would read as escaping that operator. Trailing
    whitespace is kept, so a here-document's closing line still has to repeat
    its word exactly, and a backslash followed by a space escapes the space
    rather than continuing. A space inserted where the pair was would turn the
    installer path ``scripts/install-\\`` + ``nextest.sh`` into two words the
    step never runs. Inside an ordinary single-quoted string every character
    is preserved, so a backslash-newline pair there continues the line without
    being deleted.

    Examples
    --------
    The backslashes are built with ``chr(92)``, because a docstring holding
    those escapes would end the example's line:

    >>> _continuation("cargo build " + chr(92))
    ('cargo build ', True)
    >>> _continuation("scripts/install-" + chr(92) + chr(32))[1]
    False
    >>> text, continues = _continuation("a" + chr(92) * 2)
    >>> continues
    False
    >>> text == "a" + chr(92) * 2
    True
    >>> text, continues = _continuation("a" + chr(92) * 3)
    >>> continues
    True
    >>> text == "a" + chr(92) * 2
    True
    >>> quoted = "'scripts/install-"
    >>> _continuation(quoted)[1]
    True
    >>> _continuation(quoted)[0] == quoted + chr(10)
    True
    """
    if _in_single_quotes(line):
        # The shell keeps the newline inside the quoted string.
        return f"{line}{chr(10)}", True
    run = len(line) - len(line.rstrip("\\"))
    if run % 2 == 0:
        return line, False
    return f"{line[: len(line) - run]}{chr(92) * (run - 1)}", True


def _body_step(
    documents: list[HereDocument], line: str, pending: str
) -> str:
    """Consume one physical line of the body of ``documents[0]``.

    Returns the text of an expanded body line that continues onto the next
    one, and the empty string once the line is complete. A literal body is
    compared as written: the shell expands nothing there, so a trailing
    backslash is a character and never a continuation. An expanded body has
    its backslash-newline pairs deleted before the closing word is compared,
    so ``x\\`` followed by ``EOF`` is the body line ``xEOF`` rather than the
    document's close. The closing line itself is consumed with the document,
    and a body with several documents open closes them one at a time.

    Examples
    --------
    >>> documents = [HereDocument("EOF", False, True)]
    >>> _body_step(documents, "EO" + chr(92), "")
    ''
    >>> _body_step(documents, "EOF", "")
    ''
    >>> documents
    []
    >>> expanded = [HereDocument("EOF", False, False)]
    >>> connected = _body_step(expanded, "x" + chr(92), "")
    >>> connected
    'x'
    >>> _body_step(expanded, "EOF", connected)
    ''
    >>> expanded
    [HereDocument(word='EOF', strips_tabs=False, literal=False)]
    """
    document = documents[0]
    if document.literal:
        if ends_document(document, line):
            documents.pop(0)
        return ""
    text, continues = _escape_continuation(line)
    joined = f"{pending}{text}"
    if continues:
        return joined
    if ends_document(document, joined):
        documents.pop(0)
    return ""


def _absorb(
    pending: str, line: str, lines: list[str]
) -> tuple[str, list[HereDocument]]:
    """Add one joined physical line to the command list being read.

    A comment is removed first, so a trailing comment cannot absorb the next
    line, and a blank or comment-only line between a continuation and what it
    continues is skipped as the shell skips it. A line still ending in a list
    operator waits for the next one, but the here-documents it opens start
    their bodies at that newline all the same: the shell collects them before
    it reads on, so their bodies are never part of the continued list.
    Otherwise the command line is complete: it ends the list, so it is kept
    and any here-documents it opens start their bodies on the next physical
    line.

    Examples
    --------
    >>> lines = []
    >>> _absorb("", "installer " + chr(35) + " note", lines)
    ('', [])
    >>> lines
    ['installer']
    >>> pending, documents = _absorb("", "installer &&", lines)
    >>> pending
    'installer &&'
    >>> documents
    []
    >>> _absorb(pending, "true", lines)
    ('', [])
    >>> lines
    ['installer', 'installer && true']
    >>> _absorb("", "cat <<'EOF'", [])[1]
    [HereDocument(word='EOF', strips_tabs=False, literal=True)]
    >>> _absorb("", "cat <<'EOF' &&", [])[1]
    [HereDocument(word='EOF', strips_tabs=False, literal=True)]
    """
    stripped = strip_shell_comment(line).strip()
    if pending and not stripped:
        return pending, []
    merged = f"{pending} {stripped}" if pending else stripped
    # Only this physical line can open a document the reader has not already
    # consumed: the pending text was scanned when it was read, and its bodies
    # have been dropped by now.
    documents = here_documents(stripped)
    if _ends_with_operator(merged):
        return merged, documents
    lines.append(merged)
    return "", documents


def _tail_lines(lines: list[str], pending: str, joined: str) -> list[str]:
    """Add the script's final partial line, if any, and return the result.

    A script may end without a newline, or in the middle of a list operator's
    continuation. Each is still a command line the shell would read, so the
    last physical line is absorbed once more and a list still open at the end
    is kept as the line it would have continued.

    Examples
    --------
    >>> _tail_lines([], "", "installer")
    ['installer']
    >>> _tail_lines([], "installer &&", "")
    ['installer &&']
    >>> _tail_lines(["a"], "", "")
    ['a']
    """
    if joined.strip():
        pending, _ = _absorb(pending, joined, lines)
    if pending.strip():
        lines.append(pending)
    return lines


def _has_an_open_quote(text: str) -> bool:
    """Return whether `text` ends inside a quoted string.

    The next physical line is quoted text in that case, so it must not be
    read for comments or operators of its own.

    Examples
    --------
    >>> _has_an_open_quote("echo 'a")
    True
    >>> _has_an_open_quote("echo 'a'")
    False
    >>> _has_an_open_quote("echo a")
    False
    """
    try:
        mask_quotes(text)
    except ValueError:
        return True
    return False


def _absorb_or_wait(
    joined: str, physical: str, pending: str, lines: list[str]
) -> tuple[str, str, list[HereDocument]]:
    """Join a line whose quote is open and absorb it once the quote closes.

    A trailing backslash is resolved; otherwise the newline lies inside the
    quoted text and is kept, so the three physical lines of a multi-line
    string are one command line when the quote finally closes.

    Examples
    --------
    >>> lines = []
    >>> joined, pending, documents = _absorb_or_wait("", "echo 'a", "", lines)
    >>> lines
    []
    >>> joined, pending, documents = _absorb_or_wait(joined, "c'", pending, lines)
    >>> len(lines)
    1
    """
    combined = f"{joined}{physical}"
    if _in_single_quotes(combined):
        return f"{combined}{chr(10)}", pending, []
    text, continues = _escape_continuation(physical)
    candidate = f"{joined}{text}" + ("" if continues else chr(10))
    if continues:
        return candidate, pending, []
    try:
        pending, documents = _absorb(pending, candidate, lines)
    except ValueError:
        return candidate, pending, []
    return "", pending, documents


def _command_step(
    physical: str, joined: str, pending: str, lines: list[str]
) -> tuple[str, str, list[HereDocument]]:
    """Absorb one physical command line and return the state it leaves.

    A comment's trailing backslash is comment text, so the line is stripped of
    its comment before its continuation is read. An unterminated quote means
    the shell is still reading, so the line continues either way: a trailing
    backslash is resolved, and otherwise the newline is part of the quoted
    text. A line that still continues leaves `joined` open and absorbs nothing
    yet; a complete line is absorbed and may open here-documents.

    Examples
    --------
    >>> lines = []
    >>> _command_step("installer", "", "", lines)[:2]
    ('', '')
    >>> lines
    ['installer']
    >>> _command_step("installer " + chr(92), "", "", lines)[0].endswith(" ")
    True
    >>> _command_step("# note " + chr(92), "", "", lines)[:2]
    ('', '')
    >>> lines = []
    >>> joined, _, _ = _command_step("printf x" + chr(92), "", "", lines)
    >>> _command_step("#; installer", joined, "", lines)[0]
    ''
    >>> lines[-1]
    'printf x#; installer'
    >>> _command_step("cat <<'EOF'", "", "", [])[2]
    [HereDocument(word='EOF', strips_tabs=False, literal=True)]
    >>> _command_step('echo "pre' + chr(92), "", "", [])[0]
    'echo "pre'
    >>> _command_step('pare"', "", "", [])[0].endswith(chr(10))
    True
    >>> lines = []
    >>> opened, _, _ = _command_step("echo ' #", "", "", lines)
    >>> quoted, _, _ = _command_step("installer # note", opened, "", lines)
    >>> quoted.endswith(chr(10))
    True
    >>> lines = []
    >>> opened, _, _ = _command_step("echo 'a", "", "", lines)
    >>> _command_step("b", opened, "", lines)[0].endswith(chr(10))
    True
    """
    if joined and _has_an_open_quote(joined):
        return _absorb_or_wait(joined, physical, pending, lines)
    candidate = f"{joined}{physical}"
    try:
        commented = strip_shell_comment(candidate)
    except ValueError:
        return _absorb_or_wait(joined, physical, pending, lines)
    text, continues = _continuation(commented)
    if continues:
        return text, pending, []
    try:
        pending, documents = _absorb(pending, text, lines)
    except ValueError:
        return f"{text}{chr(10)}", pending, []
    return "", pending, documents


def logical_lines(script: str) -> list[str]:
    """Return `script` as the command lines a shell would read.

    A comment is removed before its line's continuation is read, because the
    backslash that ends a comment is comment text and continues nothing; the
    shell runs the next physical line as a command of its own. Backslash
    continuations are then resolved. Each resulting line absorbs the next
    while it ends in a list operator: the shell parses the next line as that
    operator's right-hand side, so ``true ||`` followed by an installer line
    skips the installer, and an installer followed by ``&&`` carries its
    failure into the next line rather than ending as a command of its own. A here-document's body is not a command line either:
    the line that opened it names the word that ends the body, and every
    physical line up to and including that word is dropped without joining,
    as the shell reads it. The word has to repeat exactly for ``<<``, so
    `` EOF`` is body text, while a ``<<-`` document ignores leading tabs on
    both its body and its closing line.

    Examples
    --------
    >>> logical_lines("installer &&" + chr(10) + "true")
    ['installer && true']
    >>> logical_lines("true ||" + chr(10) + "installer")
    ['true || installer']
    >>> logical_lines("installer " + chr(35) + " note")
    ['installer']
    >>> logical_lines("installer " + chr(35) + " note" + chr(10) + "true")
    ['installer', 'true']
    >>> logical_lines(
    ...     "installer && " + chr(35) + " note" + chr(10) + "true"
    ... )
    ['installer && true']
    >>> logical_lines("cat <<'EOF'" + chr(10) + "installer" + chr(10) + "EOF")
    ["cat <<'EOF'"]
    >>> logical_lines(
    ...     "cat <<'EOF'" + chr(10) + "installer" + chr(10) + "EOF"
    ...     + chr(10) + "installer"
    ... )
    ["cat <<'EOF'", 'installer']
    >>> logical_lines(
    ...     "cat <<'EOF'" + chr(10) + " installer" + chr(10) + "EOF"
    ...     + chr(10) + "installer"
    ... )
    ["cat <<'EOF'", 'installer']
    >>> logical_lines(
    ...     "cat <<'EOF'" + chr(10) + "EO" + chr(92) + chr(10) + "F"
    ...     + chr(10) + "installer" + chr(10) + "EOF"
    ... )
    ["cat <<'EOF'"]
    >>> continued = "cat <<'EOF' &&" + chr(10) + "installer" + chr(10) + "EOF"
    >>> logical_lines(continued + chr(10) + "true")
    ["cat <<'EOF' && true"]
    """
    lines: list[str] = []
    documents: list[HereDocument] = []
    pending = ""
    body = ""
    joined = ""
    for physical in script.splitlines():
        if documents:
            body = _body_step(documents, physical, body)
            continue
        joined, pending, documents = _command_step(
            physical, joined, pending, lines
        )
    return _tail_lines(lines, pending, joined)