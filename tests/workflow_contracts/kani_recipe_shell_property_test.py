"""Properties of the shell reading a Kani recipe line gets.

Whether a recipe line hides a failure or puts a second command after the
checker decides what `make kani` actually gates. The quoted and escaped
shapes are the ones a text search gets wrong, so these properties generate
them rather than list them: a `#` inside quotes is not a comment, an escaped
space keeps a `#` in its word, and an operator inside quotes is text.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st

from kani_recipe_shell import comment_start, control_operators, masked

#: A word with no quoting, comment, escape or operator character.
_PLAIN = st.text(
    alphabet=string.ascii_letters + string.digits + "._-/:=,",
    min_size=1,
    max_size=10,
)

#: Words joined into one command line.
_LINE = st.lists(_PLAIN, min_size=1, max_size=4).map(" ".join)

#: One shell operator the reader refuses.
_OPERATOR = st.sampled_from((";", "&&", "&", "|"))


@given(_LINE)
def test_a_plain_command_has_no_comment_and_no_operator(line: str) -> None:
    """Nothing special is reported for a command made of plain words."""
    assert comment_start(line) is None
    assert control_operators(line) == []


@given(_LINE, _PLAIN)
def test_a_comment_begins_at_an_unquoted_word_boundary(
    line: str, note: str
) -> None:
    """`# ` after a word is a comment, and nothing after it is command text."""
    command = f"{line} # {note}"
    start = comment_start(command)
    assert start is not None
    assert command[start] == "#"


@given(_LINE, _OPERATOR)
def test_an_unquoted_operator_is_reported(line: str, operator: str) -> None:
    """A second command after an operator would hide the checker's status."""
    assert control_operators(f"{line} {operator} true") == [operator]


@given(_PLAIN, _OPERATOR)
def test_operators_inside_single_quotes_are_text(word: str, operator: str) -> None:
    """The shell does not read an operator inside quotes."""
    assert control_operators(f"'{word} {operator} {word}'") == []


@given(_PLAIN, _OPERATOR)
def test_operators_inside_double_quotes_are_text(word: str, operator: str) -> None:
    """Double quotes shield operators too."""
    assert control_operators(f'"{word} {operator} {word}"') == []


@given(_PLAIN, _PLAIN)
def test_a_hash_inside_quotes_is_not_a_comment(left: str, right: str) -> None:
    """A quoted value may contain `#` and still carry options after it."""
    command = f'--target-dir "{left} #{right}" --jobs=2'
    assert comment_start(command) is None


@given(_PLAIN, _PLAIN)
def test_an_escaped_space_keeps_a_hash_in_its_word(left: str, right: str) -> None:
    """`a\\ #b c` is one word whose second half starts with `#`."""
    command = f"{left}\\ #{right} tail"
    assert comment_start(command) is None


@given(_LINE)
def test_masking_needs_the_shells_own_suffix(line: str) -> None:
    """Only a trailing `|| true` or `|| :` swallows a failure."""
    assert masked(f"{line} || true") is True
    assert masked(f"{line} ||   :") is True
    assert masked(line) is False
    assert masked(f"{line} || false") is False
    assert masked(f"{line} | true") is False