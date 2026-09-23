"""Whether an artefact `path` entry could carry the coverage report.

Split from `coverage_boundary` because judging a path is its own grammar:
plain names, globs, and GitHub Actions expressions each need a different
reading, and each reading fails closed in its own way. The question is always
whether the report *can* leave the runner, not whether the entry is spelt like
it, so `.`, `./`, `..` and `${{ github.workspace }}` all publish it while
naming it nowhere.

Run via ``make test-workflow-contracts``.
"""

import re
import typing as typ

import pathspec

#: The report the coverage action writes, and the one CodeScene is sent.
COVERAGE_REPORT_PATH: typ.Final[str] = "lcov.info"

#: The opening and closing of a GitHub Actions expression.
_EXPRESSION: typ.Final[re.Pattern[str]] = re.compile(r"^\$\{\{(?P<body>.*)\}\}$", re.DOTALL)

#: Characters that make a path a pattern rather than a name.
_GLOB_CHARACTERS: typ.Final[frozenset[str]] = frozenset("*?[]!")

#: Where the report could sit, for testing a pattern against. The action writes
#: it at the workspace root, and a nested crate could write its own, so a
#: pattern is refused when it would match any of these rather than only the
#: first.
_REPORT_CANDIDATES: typ.Final[tuple[str, ...]] = (
    COVERAGE_REPORT_PATH,
    f"crate/{COVERAGE_REPORT_PATH}",
    f"a/b/{COVERAGE_REPORT_PATH}",
)

#: One token of an expression body: a quoted literal, an operator, or a run of
#: anything else. Literals come first so an operator inside one is not split.
_TOKEN: typ.Final[re.Pattern[str]] = re.compile(r"'(?:[^']|'')*'|\|\||&&|[^'|&]+|[|&]")

#: A quoted literal, whole.
_LITERAL: typ.Final[re.Pattern[str]] = re.compile(r"^'((?:[^']|'')*)'$")


def _pattern_could_match_the_report(entry: str) -> bool:
    """Return whether a glob entry could match the report wherever it sits."""
    # Tested against candidate locations rather than reasoned about.
    # `**/proptest-regressions/**` cannot match a report and is not an
    # offence; `**/*.info` can and is.
    spec = pathspec.PathSpec.from_lines("gitignore", [entry])
    return any(spec.match_file(candidate) for candidate in _REPORT_CANDIDATES)


def _chains(body: str) -> list[list[str]]:
    """Return an expression body as `||` alternatives of `&&` operands."""
    chains: list[list[str]] = [[]]
    operand = ""
    for token in _TOKEN.findall(body):
        if token in ("||", "&&"):
            chains[-1].append(operand.strip())
            operand = ""
            if token == "||":
                chains.append([])
        else:
            operand += token
    chains[-1].append(operand.strip())
    return chains


def _result_literal(chain: list[str]) -> str | None:
    """Return the literal a `&&` chain yields when it wins, or None if not one."""
    # `a && b && 'x'` yields its last operand when every earlier one holds, so
    # only the last operand is a value the step can receive.
    match = _LITERAL.match(chain[-1])
    return match.group(1) if match else None


def _expression_stays_outside(entry: str) -> bool:
    """Return whether every result an expression can yield is outside the tree.

    An expression is cleared only when every alternative ends in a quoted
    literal that is absolute or empty, and at least one is absolute. `${{ x ==
    'y' && '/tmp/a.log' || '' }}` is cleared; `... || github.workspace` is not,
    because that alternative yields the workspace the report sits in.
    """
    match = _EXPRESSION.match(entry)
    if match is None:
        return False
    results = [_result_literal(chain) for chain in _chains(match.group("body"))]
    return None not in results and _all_outside(typ.cast("list[str]", results))


def _all_outside(values: list[str]) -> bool:
    """Return whether literal results are all absolute or empty, one absolute."""
    return any(values) and all(not value or value.startswith("/") for value in values)


def _names_the_workspace(entry: str) -> bool:
    """Return whether a plain relative entry is the workspace or above it."""
    parts = [part for part in entry.split("/") if part not in ("", ".")]
    return not parts or ".." in parts


def could_hold_the_report(entry: str) -> bool:
    """Return whether one `path` entry could carry the coverage report.

    Examples
    --------
    >>> could_hold_the_report("./")
    True
    >>> could_hold_the_report("${{ x == 'y' && '/tmp/a.log' || '' }}")
    False
    >>> could_hold_the_report("${{ x && '/tmp/a.log' || github.workspace }}")
    True
    """
    cleaned = entry.strip()
    if not cleaned or COVERAGE_REPORT_PATH in cleaned:
        return True
    # An absolute path is somewhere other than the workspace unless it names
    # the report, which the line above has already ruled out.
    if cleaned.startswith("/"):
        return False
    if "${{" in cleaned:
        # A relative path with an expression inside it names something the
        # expression decides, and could be the workspace; so does a whole
        # expression unless every result it can yield is outside the tree.
        return not _expression_stays_outside(cleaned)
    if _GLOB_CHARACTERS & set(cleaned):
        return _pattern_could_match_the_report(cleaned)
    return _names_the_workspace(cleaned)
