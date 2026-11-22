"""Read where a shell line's quotes stand.

A logical line is assembled from physical ones, and a quote left open at the
end of one is what tells the reader the line continues: the next physical line
is quoted text, so its comment characters and operators are data. An ordinary
single-quoted string also preserves every character, including a
backslash-newline pair, while a double-quoted one lets a backslash escape the
character after it.

This module holds that scan. ``scheduling_line_reading`` uses it to join
physical lines, and the split keeps both files inside the 400-line limit
``AGENTS.md`` sets.
"""

from __future__ import annotations

from scheduling_quote_reading import mask_quotes


def in_single_quotes(text: str) -> bool:
    """Return whether `text` ends inside an ordinary single-quoted string.

    The shell preserves every character of such a string, including a
    backslash-newline pair, so a line that ends inside one continues without
    the pair being deleted.

    Examples
    --------
    >>> in_single_quotes("echo 'a")
    True
    >>> in_single_quotes('echo "a')
    False
    >>> in_single_quotes("echo 'a'")
    False
    >>> in_single_quotes('echo "" ; ' + chr(39) + "a")
    True
    """
    quote = ""
    index = 0
    while index < len(text):
        char, index = quote_step(text, index, quote)
        if quote and char == quote:
            quote = ""
        elif not quote and char in "'\"":
            quote = char
    return quote == "'"


def quote_step(text: str, index: int, quote: str) -> tuple[str, int]:
    """Return the character that decides quoting, and the index after it.

    A backslash escapes the character after it wherever it is read, so the
    pair is consumed whole and the escaped character is not returned as quote
    syntax.

    Examples
    --------
    >>> quote_step('a"b', 0, "")
    ('a', 1)
    >>> escaped = chr(92) + '"'
    >>> quote_step(escaped, 0, '"')
    ('', 2)
    """
    if quote != "'" and text[index : index + 1] == "\\":
        return "", index + 2
    return text[index : index + 1], index + 1


def has_an_open_quote(text: str) -> bool:
    """Return whether `text` ends inside a quoted string.

    The next physical line is quoted text in that case, so it must not be
    read for comments or operators of its own.

    Examples
    --------
    >>> has_an_open_quote("echo 'a")
    True
    >>> has_an_open_quote("echo 'a'")
    False
    >>> has_an_open_quote("echo a")
    False
    """
    try:
        mask_quotes(text)
    except ValueError:
        return True
    return False
