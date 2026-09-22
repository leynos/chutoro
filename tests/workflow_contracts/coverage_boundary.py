"""What a pull-request-reachable workflow may not touch.

Pull-request CI generates `lcov.info` and compares it with the ratcheted
baseline derived from `main`. It does not publish that report as an artefact,
invoke the CodeScene coverage action, run a `cs-coverage` command, name the
CodeScene host, forward every secret with `secrets: inherit`, or carry the
credential any of those needs. Those belong to `coverage-main.yml`, which is
the only writer of persistent coverage state. The rules run over every local
workflow a pull-request lane calls, not over the lane alone.

The coverage action archives the report it generated under a step of its own,
so declining that archive is part of the same boundary: a caller that reaches
the action without the opt-out has published the report whether or not the
workflow declares an artefact step. That rule is checked here rather than in
the workflow, because the action's own step is not the caller's to see.

These readers take a parsed document and its raw text rather than reading
files, so the contract beside them can drive shapes this repository does not
have. Parameterized over this repository's own workflows alone, a reader that
answered nothing would agree with a correct one exactly.

Run via ``make test-workflow-contracts``.
"""

import collections.abc as cabc
import re
import typing as typ

import pathspec

from workflow_reach import reachable_workflows
from workflow_support import jobs, run_script, steps, uses_reference

#: The action that generates coverage. A pull request calls it in ratchet mode
#: and stops there; `main` calls it to produce the report it publishes.
GENERATE_COVERAGE_ACTION: typ.Final[str] = (
    "leynos/shared-actions/.github/actions/generate-coverage"
)

#: The action that submits a report to CodeScene, in either of its modes.
#: `main` owns this call.
UPLOAD_COVERAGE_ACTION: typ.Final[str] = (
    "leynos/shared-actions/.github/actions/upload-codescene-coverage"
)

#: The generic artefact action. A pull request must not carry the report to it
#: under any step name.
PUBLISH_ARTEFACT_ACTION: typ.Final[str] = "actions/upload-artifact"

#: The input that suppresses the coverage action's own archive step, and the
#: value that suppresses it. A pull-request-reachable caller must set both, or
#: the action publishes the report this boundary exists to keep local.
PUBLICATION_OPT_OUT_INPUT: typ.Final[str] = "publish-artefact"
PUBLICATION_OPT_OUT_VALUE: typ.Final[str] = "false"

#: The credential the CodeScene upload reads. It must not appear in a workflow
#: a pull request can reach, in a parsed value or anywhere in the raw text.
CREDENTIAL_ENVIRONMENT_KEY: typ.Final[str] = "CS_ACCESS_TOKEN"

#: The command form of the same upload, which needs no action reference.
COVERAGE_COMMAND: typ.Final[str] = "cs-coverage"

#: The report the coverage action writes, and the one CodeScene is sent.
COVERAGE_REPORT_PATH: typ.Final[str] = "lcov.info"

#: The service itself. A pull-request lane that names its host is talking to
#: it by some route other than the action, which is the same dependency the
#: boundary exists to remove, spelt as a `curl`.
CODESCENE_HOST: typ.Final[str] = "codescene.io"

#: The `secrets:` value that forwards every secret the caller holds, the
#: credential included, without naming any of them.
INHERITED_SECRETS: typ.Final[str] = "inherit"


def action_of(step: dict[str, typ.Any]) -> str:
    """Return a step's action reference without its version.

    Parameters
    ----------
    step : dict[str, typ.Any]
        One parsed workflow step.

    Returns
    -------
    str
        The reference with its `@version` removed, or the empty string when the
        step runs a command rather than an action.
    """
    # Splitting on the version separator rather than matching a prefix keeps
    # `upload-codescene-coverage-legacy` from reading as the real action.
    return uses_reference(step).split("@", 1)[0]


#: The opening of a GitHub Actions expression. Its value is decided at run
#: time, so a path containing one says nothing here about what it will name.
_EXPRESSION_OPENING: typ.Final[str] = "${{"

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


def _pattern_could_match_the_report(entry: str) -> bool:
    """Return whether a glob entry could match the report wherever it sits."""
    # The pattern is tested against candidate locations rather than reasoned
    # about. `**/proptest-regressions/**` cannot match a report at the root and
    # is not an offence; `**/*.info` can and is. Refusing every glob was the
    # first draft and it condemned four honest log uploads in this repository.
    spec = pathspec.PathSpec.from_lines("gitignore", [entry])
    return any(spec.match_file(candidate) for candidate in _REPORT_CANDIDATES)


#: A single-quoted literal inside a GitHub Actions expression.
_EXPRESSION_LITERAL: typ.Final[re.Pattern[str]] = re.compile(r"'([^']*)'")


def _expression_names_somewhere_absolute(entry: str) -> bool:
    """Return whether an expression can only produce a path outside the tree."""
    # An entry whose value is decided at run time could be the workspace, and
    # the workspace holds the report. The exception is an expression that
    # chooses between absolute paths: `${{ x == 'y' && '/tmp/a.log' || '' }}`
    # cannot be the workspace whichever arm wins. A literal counts as a path
    # only when it looks like one; `'y'` above is a comparison operand.
    literals = _EXPRESSION_LITERAL.findall(entry)
    paths = [value for value in literals if "/" in value or "." in value]
    return bool(paths) and all(value.startswith("/") for value in paths)


def _could_hold_the_report(entry: str) -> bool:
    """Return whether one `path` entry could carry the coverage report."""
    # Fails closed, because the question is whether the report *can* leave the
    # runner, not whether the entry is spelt like it. A substring test for
    # `lcov.info` clears `.`, `./`, `..` and `${{ github.workspace }}`, each of
    # which uploads the workspace the report sits in.
    cleaned = entry.strip()
    if not cleaned or COVERAGE_REPORT_PATH in cleaned:
        return True
    # An absolute path is somewhere other than the workspace unless it names
    # the report, which the line above has already ruled out. The estate writes
    # the report into the workspace and refers to it relatively everywhere, so
    # `/tmp/bench.log` is not this rule's business.
    if cleaned.startswith("/"):
        return False
    if _EXPRESSION_OPENING in cleaned:
        return not _expression_names_somewhere_absolute(cleaned)
    if _GLOB_CHARACTERS & set(cleaned):
        return _pattern_could_match_the_report(cleaned)
    parts = [part for part in cleaned.split("/") if part not in ("", ".")]
    return not parts or ".." in parts


def publishes_the_coverage_report(step: dict[str, typ.Any]) -> bool:
    """Return whether a step publishes the coverage report as an artefact.

    Parameters
    ----------
    step : dict[str, typ.Any]
        One parsed workflow step.

    Returns
    -------
    bool
        True when the step uploads, or could upload, the report. A step of the
        artefact action that names no path uploads the workspace, which holds
        the generated report; so does a path of `.`, one reaching upward
        through `..`, an unresolved relative expression, or a pattern that
        matches the report.
    """
    if action_of(step) != PUBLISH_ARTEFACT_ACTION:
        return False
    with_ = step.get("with")
    if not isinstance(with_, dict) or "path" not in with_:
        return True
    declared = str(with_["path"])
    # `path` is newline-separated, and one unsafe entry publishes the report
    # whatever the others name. An empty value is the workspace.
    return not declared.strip() or any(
        _could_hold_the_report(entry) for entry in declared.splitlines()
    )


def declines_the_generated_report_archive(step: dict[str, typ.Any]) -> bool:
    """Return whether a step tells the coverage action not to archive.

    Parameters
    ----------
    step : dict[str, typ.Any]
        One parsed workflow step.

    Returns
    -------
    bool
        True when the step invokes the coverage action and passes the
        publication opt-out. The value is compared as the string the action
        itself compares against, so `false`, not a falsy stand-in, suppresses
        the upload.
    """
    if action_of(step) != GENERATE_COVERAGE_ACTION:
        return False
    with_ = step.get("with")
    if not isinstance(with_, dict):
        return False
    # Compared as the string the action itself compares against, so `false`,
    # not a falsy stand-in, is what suppresses the upload.
    return with_.get(PUBLICATION_OPT_OUT_INPUT) == PUBLICATION_OPT_OUT_VALUE


def _iter_strings(value: object) -> cabc.Iterator[str]:
    """Yield every string nested anywhere in a parsed YAML value."""
    match value:
        case str():
            yield value
        case dict():
            for key, item in value.items():
                yield from _iter_strings(key)
                yield from _iter_strings(item)
        case list():
            for item in value:
                yield from _iter_strings(item)
        case _:
            return


def _step_offences(where: str, step: dict[str, typ.Any]) -> list[str]:
    """Return every prohibited reference one step makes."""
    offences: list[str] = []
    if publishes_the_coverage_report(step):
        offences.append(f"{where} publishes the coverage report as an artefact")
    if action_of(step) == GENERATE_COVERAGE_ACTION and not (
        declines_the_generated_report_archive(step)
    ):
        offences.append(
            f"{where} invokes the coverage action without declining its own "
            f"archive ({PUBLICATION_OPT_OUT_INPUT}: {PUBLICATION_OPT_OUT_VALUE})"
        )
    if action_of(step) == UPLOAD_COVERAGE_ACTION:
        offences.append(f"{where} invokes the CodeScene coverage action")
    if COVERAGE_COMMAND in run_script(step):
        offences.append(f"{where} runs a {COVERAGE_COMMAND} command")
    return offences


def _mentions(
    name: str, document: dict[str, typ.Any], raw_text: str, needle: str
) -> list[str]:
    """Return one offence per place a workflow names the given text."""
    # The raw text is read as well as the parsed values, so a reference inside
    # a comment, or in a shape the parser flattened away, is still reported.
    # The parsed reading is what answers for a document built in a test, which
    # has no raw text to read. Both ignore case: GitHub resolves
    # `secrets.cs_access_token` to the same secret, and a host name is
    # case-insensitive.
    folded = needle.lower()
    found = [f"{name}: raw text references {needle}"] if folded in raw_text.lower() else []
    found.extend(
        f"{name}: parsed value references {needle}"
        for value in _iter_strings(document)
        if folded in value.lower()
    )
    return found


def coverage_surface_offenders(
    name: str, document: dict[str, typ.Any], raw_text: str
) -> list[str]:
    """Return every prohibited coverage-surface reference in one workflow.

    Examples
    --------
    >>> coverage_surface_offenders(
    ...     "x.yml", {"jobs": {"a": {"uses": "o/r/.github/workflows/w.yml@v1",
    ...     "secrets": "inherit"}}}, ""
    ... )
    ['x.yml:a forwards every secret (secrets: inherit), the credential included']
    """
    offenders: list[str] = []
    for job_name, definition in jobs(document).items():
        # `secrets: inherit` hands the credential to the callee without the
        # caller's text ever naming it, so neither the raw nor the parsed
        # sweep below can see it. A named forwarding is seen by both.
        if isinstance(definition, dict) and definition.get("secrets") == INHERITED_SECRETS:
            offenders.append(
                f"{name}:{job_name} forwards every secret "
                f"(secrets: {INHERITED_SECRETS}), the credential included"
            )
        for index, step in enumerate(steps(definition)):
            offenders.extend(_step_offences(f"{name}:{job_name}: step {index}", step))
    offenders.extend(_mentions(name, document, raw_text, CREDENTIAL_ENVIRONMENT_KEY))
    offenders.extend(_mentions(name, document, raw_text, CODESCENE_HOST))
    return offenders


def pull_request_offenders(
    entry: str,
    documents: cabc.Mapping[str, dict[str, typ.Any]],
    raw_texts: cabc.Mapping[str, str],
) -> list[str]:
    """Return every offence in a workflow and in each local workflow it reaches.

    Parameters
    ----------
    entry : str
        The file name of a workflow a pull request can start.
    documents : collections.abc.Mapping[str, dict[str, typing.Any]]
        Every parsed workflow in the repository, by file name.
    raw_texts : collections.abc.Mapping[str, str]
        The raw text of the same workflows, by file name.

    Returns
    -------
    list[str]
        The offences of every workflow in the entry's closure, and one for
        each local call naming a workflow that is not there to read.
    """
    # Every pull-request clause runs over the closure rather than the entry,
    # because a reusable child declares `workflow_call` alone and so reads as
    # unreachable while the pull request runs it with the caller's secrets.
    reach = reachable_workflows(entry, documents)
    offenders = [
        offence
        for reached in reach.reached
        for offence in coverage_surface_offenders(
            reached, documents[reached], raw_texts.get(reached, "")
        )
    ]
    offenders.extend(
        f"{entry} reaches a local workflow {missing} that no rule could read"
        for missing in reach.missing
    )
    return offenders
