"""Reading a ``cargo nextest list`` transcript as a set of test names.

Separated from ``dev_fast_make_support`` so the listing parse and the recipe
tokenizing stay legible apart, and so neither module outgrows the 400-line
limit ``AGENTS.md`` sets. It lives in a support module rather than a test
module so the end-to-end suite can consume it without importing another test
suite.

Nothing here compiles a workspace or shells out. The parser reads text, which
is what lets its cases be literal transcripts rather than opt-in builds.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import re

#: Colour must be disabled for a listing parse. An escape code splits the
#: binary id from the test name, and an exact suffix match would then fail
#: silently - the parse would return fewer names and a comparison built on it
#: would pass for the wrong reason.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _parse_nextest_listing(listing: str) -> set[str]:
    """Return the test names a ``nextest list -T oneline`` transcript names.

    Pure: it reads text and decides nothing about where that text came from,
    which lets the cases in ``dev_fast_listing_parse_test`` check the parse
    without compiling the workspace. A line is ``<binary-id> <test-name>``:
    exactly two whitespace fields. Requiring two, rather than requiring ``::``,
    is the whole point. An integration binary's id carries the separator
    (``pkg::suite``) while its tests need not, and a root-level unit test has
    no separator anywhere, so a ``::`` test drops real tests from *both*
    selections and the comparison then passes for the omission it exists to
    catch. The escape strip is defensive: colour is disabled for the call, but
    a coloured line would leave the reset sequence between the fields and the
    caller's suffix match would then fail silently, reading as a case the
    filter dropped.

    >>> sorted(_parse_nextest_listing("pkg root_case\\npkg::suite nested::case\\n"))
    ['pkg root_case', 'pkg::suite nested::case']
    >>> _parse_nextest_listing("pkg\\n\\n")
    set()
    """
    return {
        " ".join(fields)
        for line in listing.splitlines()
        if len(fields := ANSI_ESCAPE.sub("", line).split()) == 2
    }
