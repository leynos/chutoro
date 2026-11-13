"""Default-run cases for the development-selection comparison.

The live comparison in `dev_fast_end_to_end_test` compiles the workspace and is
opt-in. These cases decide the same question from fabricated listings, so a
comparison that stops rejecting an omission is caught on every contract run.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from dev_fast_selection_contract import (
    POISONED_LOCK_TEST,
    assert_development_selection_is_standard_minus_one,
    selected,
)

#: Three entries: two root-level cases with no module path, and the case the
#: development route must drop. Two root-level entries rather than one, so a
#: row that omits a single root-level case is distinguishable from a row that
#: omits everything.
_STANDARD = frozenset(
    {
        "chutoro-providers root_case",
        "chutoro-providers other_root_case",
        f"chutoro-core {POISONED_LOCK_TEST}",
    }
)

_INTENDED = _STANDARD - {f"chutoro-core {POISONED_LOCK_TEST}"}


@pytest.mark.parametrize(
    ("dev_selection", "is_accepted"),
    (
        pytest.param(_INTENDED, True, id="standard-minus-the-one-case"),
        # Keeps the poisoned-lock entry out, so the first two assertions pass
        # and only the whole-selection equality can reject it.
        pytest.param(
            _INTENDED - {"chutoro-providers root_case"},
            False,
            id="drops-a-root-level-case",
        ),
        pytest.param(set(), False, id="drops-everything"),
        pytest.param(set(_STANDARD), False, id="drops-nothing"),
    ),
)
def test_selection_comparison_accepts_only_the_intended_selection(
    dev_selection: set[str], is_accepted: bool
) -> None:
    """Only the exact standard-set-minus-one-case selection is accepted.

    Each rejected row omits a case the development route must run. The
    root-level row is the one a ``::``-keyed comparison would silently pass,
    which is why it is stated here rather than only in the opt-in suite. It
    keeps the poisoned-lock case out of the selection, so it clears the two
    preceding assertions and fails on the equality alone; otherwise the
    equality could be deleted and this row would still pass.

    The extra case the development side must also drop is checked by the
    identical assertion in the other direction: `drops-nothing` keeps the case
    the development route must not run.

    >>> sorted(_INTENDED)
    ['chutoro-providers other_root_case', 'chutoro-providers root_case']
    """
    if is_accepted:
        assert_development_selection_is_standard_minus_one(
            set(dev_selection), set(_STANDARD)
        )
    else:
        with pytest.raises(AssertionError):
            assert_development_selection_is_standard_minus_one(
                set(dev_selection), set(_STANDARD)
            )


def test_suffix_match_finds_a_root_level_entry() -> None:
    """An entry with no module path is matched, not skipped.

    `selected` matches by suffix so that a root-level case — whose name holds
    no ``::`` — is found. A comparison keyed on the separator would return an
    empty set here and then agree with itself about the omission.
    """
    assert selected({"chutoro-providers root_case"}, "root_case") == {
        "chutoro-providers root_case"
    }
