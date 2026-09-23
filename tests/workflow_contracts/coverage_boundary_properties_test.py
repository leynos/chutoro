"""Properties of the CV-005 readers over inputs no table enumerates.

The parameterised cases beside this module are the named boundaries. These
generate the open-ended spaces around them: artefact path lists, expressions
choosing between paths, trigger declarations, upload conditions and small
reusable-workflow graphs. Each oracle is built from how the input was
generated, never by calling the reader under test.

Run via ``make test-workflow-contracts``.
"""

import typing as typ

from coverage_boundary import publishes_the_coverage_report
from coverage_publisher import TRUNK_REF_GUARD, upload_condition_offences
from hypothesis import given
from hypothesis import strategies as st
from workflow_reach import (
    Reach,
    declared_triggers,
    local_workflows_called_by,
    reachable_workflows,
)
from workflow_support import parse_workflow_text

#: Names that cannot collide with anything the report could sit under. The
#: reader tests patterns against a few candidate directories, and a generated
#: name equal to one of those would be a genuine match, not a safe entry.
_NAMES = st.from_regex(r"x[a-z]{2,8}", fullmatch=True)

_SAFE_ENTRIES = st.one_of(
    _NAMES.map(lambda name: f"/tmp/{name}.log"),
    _NAMES.map(lambda name: f"/tmp/bench-${{{{ matrix.{name} }}}}.log"),
    _NAMES.map(lambda name: f"{name}/"),
    _NAMES.map(lambda name: f"**/{name}/**"),
)

_UNSAFE_ENTRIES = st.one_of(
    st.sampled_from(
        ["lcov.info", ".", "./", "**/*.info", "${{ github.workspace }}", "**/lcov.info"]
    ),
    _NAMES.map(lambda name: f"../{name}"),
    _NAMES.map(lambda name: f"{name}/lcov.info"),
    _NAMES.map(lambda name: f"${{{{ env.{name.upper()} }}}}"),
)

_ABSOLUTE = _NAMES.map(lambda name: f"/tmp/{name}.log")
_RELATIVE = st.one_of(
    _NAMES.map(lambda name: f"{name}/lcov.info"),
    _NAMES.map(lambda name: f"./{name}"),
    _NAMES.map(lambda name: f"{name}.log"),
)

_TRIGGERS = ("push", "pull_request", "pull_request_target", "schedule", "workflow_run")


def _artefact_step(path: str) -> dict[str, typ.Any]:
    """Return an artefact upload step declaring the given `path`."""
    return {"uses": "actions/upload-artifact@abc", "with": {"path": path}}


@given(
    safe=st.lists(_SAFE_ENTRIES, min_size=1, max_size=5),
    unsafe=st.one_of(st.none(), _UNSAFE_ENTRIES),
    position=st.integers(min_value=0, max_value=5),
)
def test_one_report_carrying_entry_condemns_the_whole_path(
    safe: list[str], unsafe: str | None, position: int
) -> None:
    """Safe entries alone pass; one unsafe entry anywhere publishes the report."""
    entries = list(safe)
    if unsafe is not None:
        entries.insert(min(position, len(entries)), unsafe)
    publishes = publishes_the_coverage_report(_artefact_step("\n".join(entries)))
    assert publishes is (unsafe is not None), entries


def _arms_with_one_relative(draw: st.DrawFn) -> tuple[list[str], bool]:
    """Draw two arms, at least one absolute, and maybe a relative replacement."""
    arms = [draw(_ABSOLUTE), draw(st.one_of(_ABSOLUTE, st.just("")))]
    relative = draw(st.one_of(st.none(), _RELATIVE))
    if relative is not None:
        arms[draw(st.integers(min_value=0, max_value=1))] = relative
    return arms, relative is not None


_ARMS = st.composite(_arms_with_one_relative)()


@given(condition=_NAMES, drawn=_ARMS)
def test_an_expression_is_cleared_only_when_every_arm_is_absolute(
    condition: str, drawn: tuple[list[str], bool]
) -> None:
    """Replacing either arm with a relative path turns a safe choice unsafe."""
    arms, has_relative = drawn
    path = f"${{{{ matrix.{condition} == 'y' && '{arms[0]}' || '{arms[1]}' }}}}"
    assert publishes_the_coverage_report(_artefact_step(path)) is has_relative, path


@given(
    names=st.lists(st.sampled_from(_TRIGGERS), min_size=1, max_size=5, unique=True),
    shape=st.sampled_from(["scalar", "sequence", "mapping"]),
    quoted=st.booleans(),
)
def test_every_trigger_shape_reads_back_the_declared_names(
    names: list[str], shape: str, *, quoted: bool
) -> None:
    """Each shape, under either key, yields exactly the names it declares."""
    if shape == "scalar":
        names = names[:1]
        body = f" {names[0]}\n"
    elif shape == "sequence":
        body = f" [{', '.join(names)}]\n"
    else:
        body = "\n" + "".join(f"  {name}:\n" for name in names)
    key = "'on'" if quoted else "on"
    document = parse_workflow_text(f"{key}:{body}jobs: {{}}\n")
    assert isinstance(document, dict)
    assert sorted(declared_triggers(document)) == sorted(names)


_CONJUNCTS = st.sampled_from(
    [
        "env.CS_ACCESS_TOKEN != ''",
        "github.event_name == 'push'",
        "success()",
        "github.repository == 'leynos/chutoro'",
    ]
)


@given(
    others=st.lists(_CONJUNCTS, max_size=3),
    guarded=st.booleans(),
    position=st.integers(min_value=0, max_value=3),
    alternative=st.one_of(st.none(), _CONJUNCTS),
)
def test_the_upload_is_confined_only_by_the_guard_as_a_conjunct(
    others: list[str], position: int, alternative: str | None, *, guarded: bool
) -> None:
    """Accepted exactly when the guard is present and no `||` follows."""
    conjuncts = list(others)
    if guarded:
        conjuncts.insert(min(position, len(conjuncts)), TRUNK_REF_GUARD)
    condition = " && ".join(conjuncts)
    if alternative is not None:
        condition = f"{condition} || {alternative}" if condition else alternative
    accepted = not upload_condition_offences(condition)
    assert accepted is (guarded and alternative is None), condition


_GRAPHS = st.integers(min_value=1, max_value=6).flatmap(
    lambda size: st.lists(
        st.lists(st.integers(min_value=0, max_value=size), max_size=3),
        min_size=size,
        max_size=size,
    )
)


def _assert_closed_under_calls(reach: Reach, documents: dict[str, dict[str, typ.Any]]) -> None:
    """Every call from a reached workflow lands in the reached or missing list."""
    for name in reach.reached:
        for called in local_workflows_called_by(documents[name]):
            assert called in reach.reached or called in reach.missing, (name, called)


def _called_by_an_earlier(name: str, earlier: list[str], documents: dict[str, typ.Any]) -> bool:
    """Return whether a workflow earlier in the walk calls this one."""
    return any(name in local_workflows_called_by(documents[caller]) for caller in earlier)


def _assert_nothing_uncalled(reach: Reach, documents: dict[str, dict[str, typ.Any]]) -> None:
    """Every reached workflow but the entry, and every missing one, was called."""
    for later, name in enumerate(reach.reached[1:], start=1):
        assert _called_by_an_earlier(name, reach.reached[:later], documents), name
    for name in reach.missing:
        assert name not in documents, name
        assert _called_by_an_earlier(name, reach.reached, documents), name


@given(edges=_GRAPHS)
def test_the_walk_reaches_exactly_what_is_called_and_reads_it_once(
    edges: list[list[int]],
) -> None:
    """The reached list is closed under calls and holds nothing uncalled.

    Node `len(edges)` is never defined, so calls to it stand for a local call
    naming a file that is not there.
    """
    names = [f"w{index}.yml" for index in range(len(edges) + 1)]
    documents = {
        names[index]: {
            "jobs": {
                f"j{order}": {"uses": f"./.github/workflows/{names[target]}"}
                for order, target in enumerate(targets)
            }
        }
        for index, targets in enumerate(edges)
    }
    reach = reachable_workflows(names[0], documents)
    assert reach.reached[0] == names[0]
    assert len(reach.reached) == len(set(reach.reached))
    _assert_closed_under_calls(reach, documents)
    _assert_nothing_uncalled(reach, documents)
