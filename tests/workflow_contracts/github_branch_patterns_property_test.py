"""Properties of GitHub's branch-filter matching.

A required job's trigger reaches a pull request only if its branch filter does,
so the matcher decides whether a required check exists at all. The example
tables in :mod:`required_checks_test` pin the spellings that matter; these
properties state the rules over generated branches and pattern lists, where a
matcher that only happens to work on the examples would be caught.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st

from github_branch_patterns import branch_pattern_regex, selected_by

#: A branch name from characters no glob pattern reads specially.
_LITERAL_BRANCH = st.text(
    alphabet=string.ascii_lowercase + string.digits + "/_-",
    min_size=1,
    max_size=12,
)


@given(_LITERAL_BRANCH)
def test_an_empty_pattern_list_selects_nothing(branch: str) -> None:
    """No patterns can select no branches."""
    assert selected_by([], branch) is False


@given(_LITERAL_BRANCH)
def test_a_literal_pattern_matches_only_itself(branch: str) -> None:
    """With no glob metacharacter, the pattern is literal membership.

    `*`, `?`, `+` and `[` are the only characters the translator reads
    specially, so a name made of the others must match exactly itself.
    """
    assert branch_pattern_regex(branch).fullmatch(branch) is not None


@given(_LITERAL_BRANCH, _LITERAL_BRANCH)
def test_a_literal_pattern_does_not_match_another_branch(
    branch: str, other: str
) -> None:
    """Two different literal names never select each other."""
    if branch != other:
        assert branch_pattern_regex(branch).fullmatch(other) is None


@given(_LITERAL_BRANCH, _LITERAL_BRANCH)
def test_a_final_negative_pattern_deselects(branch: str, other: str) -> None:
    """A trailing `!pattern` removes what an earlier pattern selected.

    The negative pattern is guaranteed not to name `other`, so the branch is
    selected only when it is the one the positive pattern named.
    """
    patterns = [branch, f"!{branch}"]
    assert selected_by(patterns, branch) is False
    assert selected_by(patterns, other) is False


@given(_LITERAL_BRANCH)
def test_a_repeated_selection_after_an_exclusion_wins(branch: str) -> None:
    """Order matters: a later positive pattern selects the branch again."""
    assert selected_by([f"!{branch}", branch], branch) is True


@given(_LITERAL_BRANCH, _LITERAL_BRANCH)
def test_one_matching_pattern_is_enough(branch: str, other: str) -> None:
    """A list selects when any positive pattern matches, exclusions aside."""
    if branch != other:
        assert selected_by([other, branch], branch) is True
        assert selected_by([other, f"!{other}"], branch) is False