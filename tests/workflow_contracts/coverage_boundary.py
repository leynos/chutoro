"""What a pull-request-reachable workflow may not touch.

Pull-request CI generates `lcov.info` and compares it with the ratcheted
baseline derived from `main`. It does not publish that report as an artefact,
invoke the CodeScene coverage action, run a `cs-coverage` command, or carry the
credential either of those needs. Those belong to `coverage-main.yml`, which is
the only writer of persistent coverage state.

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
import typing as typ

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

PULL_REQUEST_TRIGGER: typ.Final[str] = "pull_request"

#: The variant that runs in the base repository's context and therefore *can*
#: read its secrets, unlike `pull_request`. A coverage step here would be worse
#: than one in an ordinary pull-request job, not equivalent to it.
PULL_REQUEST_TARGET_TRIGGER: typ.Final[str] = "pull_request_target"

#: The trigger that resumes a run with the base repository's privileges.
SUBMISSION_TRIGGER: typ.Final[str] = "workflow_run"

REACHABLE_TRIGGERS: typ.Final[tuple[str, ...]] = (
    PULL_REQUEST_TRIGGER,
    PULL_REQUEST_TARGET_TRIGGER,
    SUBMISSION_TRIGGER,
)


def action_of(step: dict[str, typ.Any]) -> str:
    """Return a step's action reference without its version."""
    # Splitting on the version separator rather than matching a prefix keeps
    # `upload-codescene-coverage-legacy` from reading as the real action.
    return uses_reference(step).split("@", 1)[0]


def declares_trigger(document: dict[str, typ.Any], trigger: str) -> bool:
    """Return whether a parsed workflow declares the given trigger."""
    # PyYAML reads a bare `on:` key as the boolean True, and `on:` accepts a
    # scalar, a sequence or a mapping. All four shapes answer the same
    # question, and a reader that knew only the mapping would call a
    # `on: pull_request` workflow unreachable.
    declared = document.get("on", document.get(True))
    match declared:
        case str():
            return declared == trigger
        case list():
            return trigger in declared
        case dict():
            return trigger in declared
        case _:
            return False


def is_reachable_by_a_pull_request(document: dict[str, typ.Any]) -> bool:
    """Return whether a pull request can cause this workflow to run."""
    # All three count. `pull_request_target` and `workflow_run` resume in the
    # base repository's context, so a coverage step under either reads a
    # credential in a run a pull request's contents influenced.
    return any(declares_trigger(document, name) for name in REACHABLE_TRIGGERS)


def publishes_the_coverage_report(step: dict[str, typ.Any]) -> bool:
    """Return whether a step publishes the coverage report as an artefact."""
    if action_of(step) != PUBLISH_ARTEFACT_ACTION:
        return False
    # A step of the artefact action that names no path uploads the workspace,
    # which holds the generated report, so it fails closed rather than reading
    # as an exemption.
    with_ = step.get("with")
    if not isinstance(with_, dict) or "path" not in with_:
        return True
    return COVERAGE_REPORT_PATH in str(with_["path"])


def declines_the_generated_report_archive(step: dict[str, typ.Any]) -> bool:
    """Return whether a step tells the coverage action not to archive."""
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


def coverage_surface_offenders(
    name: str, document: dict[str, typ.Any], raw_text: str
) -> list[str]:
    """Return every prohibited coverage-surface reference in one workflow."""
    offenders: list[str] = []
    for job_name, definition in jobs(document).items():
        for index, step in enumerate(steps(definition)):
            offenders.extend(_step_offences(f"{name}:{job_name}: step {index}", step))
    # The raw text is read as well as the parsed values, so a reference inside
    # a comment, or in a shape the parser flattened away, is still reported.
    if CREDENTIAL_ENVIRONMENT_KEY in raw_text:
        offenders.append(f"{name}: raw text references {CREDENTIAL_ENVIRONMENT_KEY}")
    offenders.extend(
        f"{name}: parsed value references {CREDENTIAL_ENVIRONMENT_KEY}"
        for value in _iter_strings(document)
        if CREDENTIAL_ENVIRONMENT_KEY in value
    )
    return offenders
