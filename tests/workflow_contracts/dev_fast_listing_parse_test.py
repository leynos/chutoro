"""Contracts for reading a ``cargo nextest list`` transcript.

Separated from ``dev_fast_end_to_end_test`` so the pure text parse and its
cases stay legible on their own, and so neither module outgrows the 400-line
limit ``AGENTS.md`` sets. Nothing here compiles a workspace or shells out: the
parser reads text, so every case is a literal transcript.

The distinction these contracts pin is the one the end-to-end comparison
depends on. ``nextest list -T oneline`` writes ``<binary-id> <test-name>``, and
only the *binary* id is guaranteed to carry a ``::`` separator. A root-level
unit test appears as ``chutoro-providers root_case`` with no separator at all,
so a parser that treats ``::`` as the mark of a test silently drops it from both
sides of a comparison and lets the omission the comparison exists to catch pass
unseen.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import pytest
from dev_fast_listing_support import _parse_nextest_listing

#: The full ``chutoro-core`` name of the poisoned-lock regression, as a real
#: ``cargo nextest list`` writes it: the binary prefix, then the test path.
_POISONED_LOCK_LISTING = (
    "chutoro-core hnsw::cpu::test_helpers::tests::"
    "graph_helpers_report_poisoning_without_reconfiguring"
)
#: The same line as nextest writes it under ``--color always``. The reset
#: sequence follows the test name, so an unstripped parse yields a name no
#: suffix match can find, and the exclusion would read as applied when the
#: filter had dropped nothing.
_COLOURED_POISONED_LOCK_LISTING = (
    "\x1b[35;1mchutoro-core\x1b[0m "
    "\x1b[36mhnsw::cpu::test_helpers::tests\x1b[0m\x1b[36m::\x1b[0m"
    "\x1b[34;1mgraph_helpers_report_poisoning_without_reconfiguring\x1b[0m"
)


@pytest.mark.parametrize(
    ("listing", "expected"),
    (
        pytest.param(
            _POISONED_LOCK_LISTING + "\n",
            {_POISONED_LOCK_LISTING},
            id="plain-line-keeps-the-binary-prefix",
        ),
        pytest.param(
            _COLOURED_POISONED_LOCK_LISTING + "\n",
            {_POISONED_LOCK_LISTING},
            id="coloured-line-strips-escapes",
        ),
        pytest.param(
            "chutoro-core a::b\nchutoro-core c::d\n",
            {"chutoro-core a::b", "chutoro-core c::d"},
            id="several-lines-keep-every-name",
        ),
        pytest.param(
            "chutoro-core a::b\nchutoro-core a::b\n",
            {"chutoro-core a::b"},
            id="repeated-name-collapses",
        ),
        # A root-level unit test has no module path, so the line carries no
        # ``::`` at all. Dropping it removes it from both selections, and the
        # comparison then passes for the very omission it exists to detect.
        pytest.param(
            "chutoro-providers root_case\n",
            {"chutoro-providers root_case"},
            id="root-level-test-without-a-separator-survives",
        ),
        pytest.param(
            "chutoro-compile-contracts::error root_case\n",
            {"chutoro-compile-contracts::error root_case"},
            id="separator-in-the-binary-id-is-not-the-test-name",
        ),
        # The parse sees only text, so it cannot know a line is a test except
        # by its shape. A line naming no test must not enter the set: counting
        # one would make the comparison differ for a reason the filter did not
        # cause, and the failure would point at the exclusion rather than at
        # the parse.
        pytest.param("chutoro-core\n", set(), id="line-without-a-test-names-none"),
        pytest.param("a b c\n", set(), id="extra-field-names-no-single-test"),
        pytest.param("\n\n", set(), id="blank-lines-name-no-test"),
        pytest.param("", set(), id="empty-transcript-names-no-test"),
    ),
)
def test_listing_parse_reports_test_names(listing: str, expected: set[str]) -> None:
    """The parse must report names, not lines, and must survive colour.

    Each row is one shape the transcript can take. The assertion reads the
    parsed set directly, which is what the selection comparison consumes, so a
    parser that stopped at the first line or kept the escape codes fails here
    rather than in an opt-in end-to-end run.
    """
    assert _parse_nextest_listing(listing) == expected
