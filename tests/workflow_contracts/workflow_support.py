"""Shared helpers for the workflow contract tests.

The runner-placement, tool-install, and cache-ownership contracts all parse
the same eight workflow files, so the parsing and the small vocabulary they
share (what counts as a GitHub-hosted label, which shared action owns which
cache path) live here rather than being restated in each module.
"""

from __future__ import annotations

import re
import typing as typ
from pathlib import Path

import yaml

from strict_yaml import parse_workflow_text

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
ACTIONLINT_CONFIG = ROOT / ".github" / "actionlint.yaml"

#: actions/cache v6.1.0. Ubicloud's transparent cache proxy is confirmed to
#: intercept this version's traffic, which is what makes the deprecated
#: ubicloud/cache fork unnecessary. That is a compatibility fact about a
#: specific release, not a floating preference, so this pin is asserted by
#: value: a Dependabot bump must be revalidated against the proxy and this
#: constant updated deliberately. See the developers guide, "Workflow pins
#: and Dependabot".
CACHE_ACTION_SHA = "55cc8345863c7cc4c66a329aec7e433d2d1c52a9"

#: Labels served by GitHub's own hosted runner pool.
GITHUB_HOSTED_LABELS = frozenset(
    {
        "ubuntu-latest",
        "ubuntu-24.04",
        "ubuntu-22.04",
        "windows-latest",
        "macos-latest",
    }
)

#: Cache paths each shared action owns when its cache is left enabled. A
#: caller that also declares one of these paths would give it two owners
#: with competing keys, so the contract tests check both sides together.
SHARED_ACTION_OWNED_PATHS: typ.Final[dict[str, tuple[str, ...]]] = {
    "setup-rust": ("~/.cargo/registry", "~/.cargo/git", "~/.cache/uv"),
    "generate-coverage": (
        "~/.cargo/registry",
        "~/.cargo/git",
        "~/.cargo/bin/cargo-binstall",
        "~/.cargo/bin/cargo-llvm-cov",
        "~/.cargo/bin/cargo-nextest",
    ),
    "install-whitaker": (
        "~/.cargo/bin/whitaker-installer",
        "~/.local/share/whitaker",
    ),
}


#: GitHub accepts either extension for a workflow file. Matching only one
#: would let a workflow escape every contract below without failing a test.
WORKFLOW_SUFFIXES = ("*.yml", "*.yaml")


class WorkflowReadError(OSError):
    """Raised when a workflow file or directory cannot be read.

    Reading a workflow is fallible in ways that look nothing alike from
    the caller: the directory may be absent, a file may be missing, its
    bytes may not decode, or its text may not be YAML. Letting each
    raise its own type made the filesystem boundary invisible in the
    readings above it, which reported an unreadable file with whatever
    exception the failure happened to produce. One type, naming the
    path, is what lets a contract say it could not read a workflow
    rather than fail somewhere unrelated.
    """


def workflow_paths() -> list[Path]:
    """Return every workflow file, sorted for stable test identifiers."""
    return sorted(
        path for suffix in WORKFLOW_SUFFIXES for path in WORKFLOW_DIR.glob(suffix)
    )


def read_workflow_text(path: Path) -> str:
    """Return one workflow file's raw text, or raise `WorkflowReadError`.

    For contracts that must see what the parser drops, such as a reference
    inside a comment, and that still need an unreadable file to name itself.
    """
    # The failures are converted here rather than left to the caller: a
    # query that reads the filesystem is where the contract loses the
    # ability to name the file at fault.
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        message = f"{path} could not be read: {error}"
        raise WorkflowReadError(message) from error
    except UnicodeDecodeError as error:
        message = f"{path} is not UTF-8 text: {error}"
        raise WorkflowReadError(message) from error


def read_workflow_document(path: Path) -> object:
    """Return one workflow file's parsed document, or raise."""
    text = read_workflow_text(path)
    try:
        return parse_workflow_text(text)
    except yaml.YAMLError as error:
        message = f"{path} is not YAML: {error}"
        raise WorkflowReadError(message) from error


def all_workflow_documents(
    directory: Path = WORKFLOW_DIR,
) -> dict[str, dict[str, typ.Any]]:
    """Return every workflow document in a directory, keyed by file name."""
    # `Path.glob` yields nothing for a missing path and for a path that
    # is not a directory, so without this guard the reading would return
    # an empty mapping and every contract above it would pass having
    # read no workflow at all.
    if not directory.is_dir():
        message = f"{directory} is not a directory, so no workflow was read"
        raise WorkflowReadError(message)
    paths = sorted(
        path for suffix in WORKFLOW_SUFFIXES for path in directory.glob(suffix)
    )
    documents: dict[str, dict[str, typ.Any]] = {}
    for path in paths:
        parsed = read_workflow_document(path)
        if isinstance(parsed, dict):
            documents[path.name] = parsed
    return documents


def workflow_names() -> list[str]:
    """Return every workflow file name, for parametrization."""
    return [path.name for path in workflow_paths()]


def load_workflow(name: str) -> dict[str, typ.Any]:
    """Parse one workflow file into a mapping."""
    workflow = parse_workflow_text((WORKFLOW_DIR / name).read_text(encoding="utf-8"))
    if not isinstance(workflow, dict):
        msg = f"{name} must parse to a mapping"
        raise AssertionError(msg)
    return workflow


def triggers(workflow: dict[str, typ.Any]) -> dict[str, typ.Any]:
    """Return the ``on:`` mapping (PyYAML parses a bare ``on`` as True)."""
    found = workflow.get("on", workflow.get(True))
    if not isinstance(found, dict):
        msg = "the workflow must declare an on: mapping"
        raise AssertionError(msg)
    return found


def jobs(workflow: dict[str, typ.Any]) -> dict[str, dict[str, typ.Any]]:
    """Return the workflow's jobs mapping."""
    found = workflow.get("jobs")
    if not isinstance(found, dict):
        msg = "the workflow must declare jobs"
        raise AssertionError(msg)
    return found


def job(workflow_name: str, job_name: str) -> dict[str, typ.Any]:
    """Load one named job from a workflow."""
    found = jobs(load_workflow(workflow_name)).get(job_name)
    if not isinstance(found, dict):
        msg = f"{workflow_name} must declare {job_name}"
        raise AssertionError(msg)
    return found


def steps(job_definition: dict[str, typ.Any]) -> list[dict[str, typ.Any]]:
    """Return a job's steps, or an empty list for a reusable-workflow call."""
    found = job_definition.get("steps", [])
    return [step for step in found if isinstance(step, dict)]


def runner_labels(job_definition: dict[str, typ.Any]) -> list[str]:
    """Return a job's ``runs-on`` labels as a list."""
    runs_on = job_definition.get("runs-on")
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [label for label in runs_on if isinstance(label, str)]
    return []


#: The guard that keeps a scheduled run of a lane off a paid runner.
PULL_REQUEST_EVENT_GUARD = "pull-request event"

#: The guard that keeps a fork's pull request off a paid runner. A fork
#: cannot obtain one at all, so the lane would not start without it.
NON_FORK_GUARD = "non-fork head"

#: The whole of a conditional ``runs-on``: one condition, then the label
#: taken when it holds and the label taken when it does not. Parsing the
#: outer shape and the condition separately is what lets the contracts ask
#: which guards a lane carries rather than merely whether it is paid.
CONDITIONAL_RUNS_ON = re.compile(
    r"^\$\{\{\s*(?P<condition>.+?)\s*"
    r"&&\s*'(?P<when_true>[^']+)'\s*"
    r"\|\|\s*'(?P<when_false>[^']+)'\s*\}\}$"
)

#: The fork test, written the one way the estate writes it.
_FORK_TEST = "github.event.pull_request.head.repo.fork"

#: Each condition atom the contracts recognize, and the guard it supplies.
#: Exact strings, not patterns: a looser match would accept a condition
#: that tests a sibling field and report it as the guard it is not.
_GUARD_ATOMS: typ.Final[dict[str, str]] = {
    "github.event_name == 'pull_request'": PULL_REQUEST_EVENT_GUARD,
    f"!{_FORK_TEST}": NON_FORK_GUARD,
}


class ConditionalRunner(typ.NamedTuple):
    """The two labels a conditional ``runs-on`` chooses between."""

    paid: str
    otherwise: str
    guards: frozenset[str]


def _parse_guards(condition: str) -> frozenset[str] | None:
    """Return the guards a positive condition supplies, or ``None``."""
    atoms = [atom.strip() for atom in condition.split("&&")]
    guards = {_GUARD_ATOMS.get(atom) for atom in atoms}
    if None in guards:
        return None
    return frozenset(typ.cast("set[str]", guards))


def conditional_runner(
    job_definition: dict[str, typ.Any],
) -> ConditionalRunner | None:
    """Return the labels and guards of a conditional ``runs-on``.

    Two shapes are recognized, and only two. A lane that runs only on pull
    requests names the fork test first, because the fork is the case that
    must fall back; a lane that also serves a schedule names the positive
    conditions first and joins them with ``&&``. Both are normalized here
    to the label a non-fork pull request takes, the label everything else
    takes, and the guards the condition actually supplies, so a contract
    can require a guard rather than infer it from the shape.

    Anything else returns ``None``, including a condition that tests a
    sibling field. That narrowness is the point: a parser that accepted
    any condition would report its arms as though these guards had chosen
    them.

    Parameters
    ----------
    job_definition : dict[str, typing.Any]
        One job, as parsed from a workflow document.

    Returns
    -------
    ConditionalRunner | None
        The paid label, the fallback label and the guards; ``None`` when
        ``runs-on`` names a label outright or uses an unrecognized
        condition, so the caller can fall back to :func:`runner_labels`.

    Examples
    --------
    A lane that only ever runs on a pull request guards the fork alone:

    >>> conditional_runner(
    ...     {
    ...         "runs-on": "${{ github.event.pull_request.head.repo.fork"
    ...         " && 'ubuntu-latest' || 'ubicloud-standard-2' }}"
    ...     }
    ... )
    ConditionalRunner(paid='ubicloud-standard-2', otherwise='ubuntu-latest', guards=frozenset({'non-fork head'}))

    A lane that also serves a cron guards both, and the guards are what a
    contract reads:

    >>> both = conditional_runner(
    ...     {
    ...         "runs-on": "${{ github.event_name == 'pull_request'"
    ...         " && !github.event.pull_request.head.repo.fork"
    ...         " && 'ubicloud-standard-2' || 'ubuntu-latest' }}"
    ...     }
    ... )
    >>> both.paid, sorted(both.guards)
    ('ubicloud-standard-2', ['non-fork head', 'pull-request event'])

    A plain label, or a condition on some other field, is not one of these:

    >>> conditional_runner({"runs-on": "ubuntu-latest"}) is None
    True

    Neither is a declaration whose expression carries a line break, which a
    folded scalar produces when its continuation is indented deeper than
    its key. The break is written with ``chr(10)`` rather than an escape,
    because a docstring holding the escape would end this example's line:

    >>> broken = (
    ...     "${{ github.event.pull_request.head.repo.fork"
    ...     + chr(10)
    ...     + "&& 'ubuntu-latest' || 'ubicloud-standard-2' }}"
    ... )
    >>> conditional_runner({"runs-on": broken}) is None
    True
    """
    runs_on = job_definition.get("runs-on")
    if not isinstance(runs_on, str) or "\n" in runs_on.strip():
        # A value carrying a line break reads as no conditional runner at
        # all, so `test_no_runs_on_declaration_carries_a_line_break`
        # reports it rather than this reader parsing through it. The
        # pattern's inter-token whitespace class would otherwise absorb
        # the break and return the right labels and the right guard for a
        # declaration nobody meant to write.
        return None
    match = CONDITIONAL_RUNS_ON.match(runs_on.strip())
    if match is None:
        return None
    condition = match.group("condition")
    when_true = match.group("when_true")
    when_false = match.group("when_false")
    if condition == _FORK_TEST:
        return ConditionalRunner(when_false, when_true, frozenset({NON_FORK_GUARD}))
    guards = _parse_guards(condition)
    if guards is None:
        return None
    return ConditionalRunner(when_true, when_false, guards)


def is_reusable_call(job_definition: dict[str, typ.Any]) -> bool:
    """Report whether the job delegates to a reusable workflow."""
    return "uses" in job_definition


def is_pull_request_only(job_definition: dict[str, typ.Any]) -> bool:
    """Report whether a job's condition restricts it to pull requests."""
    condition = job_definition.get("if")
    if not isinstance(condition, str):
        return False
    return "github.event_name == 'pull_request'" in condition


def declared_cache_paths(step: dict[str, typ.Any]) -> list[str]:
    """Return the paths a cache step declares, one per line."""
    with_block = step.get("with")
    if not isinstance(with_block, dict):
        return []
    path = with_block.get("path")
    if not isinstance(path, str):
        return []
    return [line.strip() for line in path.splitlines() if line.strip()]


def uses_reference(step: dict[str, typ.Any]) -> str:
    """Return a step's ``uses`` value, or the empty string."""
    reference = step.get("uses")
    return reference if isinstance(reference, str) else ""


def run_script(step: dict[str, typ.Any]) -> str:
    """Return a step's ``run`` script, or the empty string."""
    script = step.get("run")
    return script if isinstance(script, str) else ""


def shared_action_name(reference: str) -> str | None:
    """Return the shared action's directory name, if the step uses one."""
    prefix = "leynos/shared-actions/.github/actions/"
    if not reference.startswith(prefix):
        return None
    return reference.removeprefix(prefix).split("@", 1)[0]


def cache_is_enabled(step: dict[str, typ.Any]) -> bool:
    """Report whether a shared action step left its own cache enabled."""
    with_block = step.get("with")
    if not isinstance(with_block, dict):
        return True
    if with_block.get("cache-provider") == "external":
        return False
    return with_block.get("enable-cache") is not False
