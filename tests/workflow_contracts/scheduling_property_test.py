"""Properties the scheduling readers must hold over generated input.

The readers are lexical: they claim that masking sets quoted text aside, that
comment stripping removes only a comment, that joining logical lines is
stable, and that a word list survives a second redaction. Hand-picked cases
pin the shapes those claims were written for; these properties drive the same
readings over generated input, where a rule that only happens to work on the
examples would be caught.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st
from makefile_reading import _target_names
from scheduling_comment_reading import strip_shell_comment
from scheduling_line_reading import logical_lines
from scheduling_quote_reading import mask_quotes
from scheduling_wrapper_reading import argument_words

#: Text with no character the shell reads specially, so every reading is
#: expected to leave it alone.
_PLAIN = st.text(
    alphabet=string.ascii_letters + string.digits + " .,:/_-",
    max_size=12,
)

#: One shell word, with no quoting, comment character or redirection.
_WORD = st.text(
    alphabet=string.ascii_letters + string.digits + "._-/",
    min_size=1,
    max_size=6,
)

#: One simple command line.
_COMMAND = st.lists(_WORD, min_size=1, max_size=3).map(" ".join)


@st.composite
def _scripts(draw) -> str:
    """Return a script of generated command lines and list operators."""
    count = draw(st.integers(min_value=1, max_value=4))
    commands = draw(st.lists(_COMMAND, min_size=count, max_size=count))
    operators = draw(
        st.lists(
            st.sampled_from([" ; ", " && ", " || ", " | ", " & "]),
            min_size=count - 1,
            max_size=count - 1,
        )
    )
    script = commands[0]
    for operator, command in zip(operators, commands[1:]):
        script = f"{script}{operator}{command}"
    return script


@given(_PLAIN)
def test_masking_leaves_plain_text_unchanged(text: str) -> None:
    """Text with no quote or escape character is its own masked form."""
    masked = mask_quotes(text)
    assert masked.text == text
    assert masked.quoted == []


@given(_PLAIN)
def test_a_trailing_comment_is_removed(line: str) -> None:
    """A comment after a word start is removed and the rest is kept."""
    assert strip_shell_comment(f"{line} # note") == line.rstrip()


@given(_PLAIN)
def test_comment_stripping_is_idempotent(line: str) -> None:
    """Removing a line's comment twice removes no further text."""
    once = strip_shell_comment(line)
    assert strip_shell_comment(once) == once


@given(_scripts())
def test_logical_lines_are_stable(script: str) -> None:
    """Re-joining resolved logical lines yields the same command lines."""
    joined = logical_lines(script)
    assert logical_lines("\n".join(joined)) == joined


@given(st.lists(_WORD, max_size=5))
def test_argument_words_is_idempotent(words: list[str]) -> None:
    """A word list with no redirection survives a second redaction."""
    once = argument_words(words)
    assert argument_words(once) == once


@given(st.lists(st.text(alphabet="abc -", max_size=6), max_size=4))
def test_target_names_round_trip(names: list[str]) -> None:
    """Escaped spaces keep generated names apart and restore them whole."""
    kept = [name for name in names if name.strip(" ") and name == name.strip()]
    escaped = " ".join(name.replace(" ", "\\ ") for name in kept)
    assert _target_names(escaped) == kept