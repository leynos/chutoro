"""The whole-run budget, resolved the way nextest resolves it.

``property-heavy`` declares no ``global-timeout`` and inherits one through
``ci``. A contract that reads only the selected profile's own table would
report no budget for the weekly lane, and every comparison against that
budget would then be made against nothing. These drive the reading past the
configurations this repository happens to contain, so each branch of the
walk is exercised rather than only the one the tree reaches today.

See "Dense SIMD parity suite" in ``docs/developers-guide.md``.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from nextest_durations import _seconds
from profile_inheritance import ProfileInheritanceError, resolved_global_timeout

#: Durations the generator may place on a profile. The values are arbitrary;
#: the reading converts them with the same parser as the configuration file.
_DURATIONS = st.sampled_from(["1m", "2m", "30m", "80m", "1h"])


@pytest.mark.parametrize(
    ("config_text", "profile", "expected"),
    [
        pytest.param(
            '[profile.default]\nglobal-timeout = "80m"\n',
            "default",
            4800.0,
            id="the-default-profile-owns-its-budget",
        ),
        pytest.param(
            '[profile.default]\nglobal-timeout = "80m"\n'
            '[profile.ci]\ninherits = "default"\n',
            "ci",
            4800.0,
            id="an-explicit-inherits-link",
        ),
        pytest.param(
            '[profile.default]\nglobal-timeout = "80m"\n'
            '[profile.ci]\ninherits = "default"\n'
            '[profile.heavy]\ninherits = "ci"\n',
            "heavy",
            4800.0,
            id="a-chain-through-an-intermediate-profile",
        ),
        pytest.param(
            '[profile.default]\nglobal-timeout = "80m"\n[profile.local]\n',
            "local",
            4800.0,
            id="a-custom-profile-with-no-parent-still-inherits-default",
        ),
        pytest.param(
            '[profile.default]\nglobal-timeout = "20m"\n'
            '[profile.ci]\ninherits = "default"\nglobal-timeout = "40m"\n',
            "ci",
            2400.0,
            id="the-nearest-declaration-wins",
        ),
    ],
)
def test_a_profile_resolves_the_global_timeout_of_its_chain(
    config_text: str, profile: str, expected: float
) -> None:
    """The nearest ``global-timeout`` on the chain is the one nextest applies.

    Parameters
    ----------
    config_text : str
        A nextest configuration shaped as the runner accepts it.
    profile : str
        The profile whose resolved budget is wanted.
    expected : float
        The expected number of seconds.
    """
    assert resolved_global_timeout(config_text, profile) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("config_text", "profile"),
    [
        pytest.param(
            '[profile.default]\nglobal-timeout = "80m"\n'
            '[profile.a]\ninherits = "b"\n'
            '[profile.b]\ninherits = "a"\n',
            "a",
            id="a-cycle",
        ),
        pytest.param(
            '[profile.default]\nglobal-timeout = "80m"\n'
            '[profile.ci]\ninherits = "ci"\n',
            "ci",
            id="a-self-reference",
        ),
        pytest.param(
            '[profile.default]\nglobal-timeout = "80m"\n',
            "missing",
            id="an-undeclared-profile",
        ),
        pytest.param(
            "[profile.default]\n",
            "default",
            id="a-chain-with-no-budget",
        ),
    ],
)
def test_an_unresolvable_chain_is_refused(config_text: str, profile: str) -> None:
    """A chain that resolves no budget must fail rather than read as unbounded.

    Parameters
    ----------
    config_text : str
        A nextest configuration shaped as the runner accepts it.
    profile : str
        The profile whose resolution is expected to fail.
    """
    with pytest.raises(ProfileInheritanceError):
        resolved_global_timeout(config_text, profile)


@given(
    budgets=st.lists(st.none() | _DURATIONS, min_size=1, max_size=6),
    default_budget=st.none() | _DURATIONS,
)
def test_the_nearest_declared_budget_wins_on_any_chain(
    budgets: list[str | None], default_budget: str | None
) -> None:
    """A generated chain resolves the first budget it reaches.

    The chain is ``p0 -> p1 -> ... -> default``, with each profile's budget
    independently present or absent. The expected value is the first present
    budget in that order, so the property separates the nearest-declaration
    rule from any particular chain the repository happens to contain.
    """
    names = [f"p{index}" for index in range(len(budgets))]
    lines: list[str] = []
    for index, name in enumerate(names):
        parent = names[index + 1] if index + 1 < len(names) else "default"
        lines.append(f"[profile.{name}]")
        lines.append(f'inherits = "{parent}"')
        if budgets[index] is not None:
            lines.append(f'global-timeout = "{budgets[index]}"')
    lines.append("[profile.default]")
    if default_budget is not None:
        lines.append(f'global-timeout = "{default_budget}"')
    config_text = "\n".join(lines)

    expected = next(
        (budget for budget in [*budgets, default_budget] if budget is not None),
        None,
    )
    if expected is None:
        with pytest.raises(ProfileInheritanceError):
            resolved_global_timeout(config_text, names[0])
        return
    assert resolved_global_timeout(config_text, names[0]) == pytest.approx(
        _seconds(expected)
    )


@given(length=st.integers(min_value=1, max_value=6))
def test_any_inheritance_cycle_is_refused(length: int) -> None:
    """A generated cycle is refused rather than resolved through ``default``.

    ``length == 1`` is the self-reference case, which the pinned nextest
    rejects exactly as it rejects a longer cycle.
    """
    names = [f"p{index}" for index in range(length)]
    lines: list[str] = []
    for index, name in enumerate(names):
        lines.append(f"[profile.{name}]")
        lines.append(f'inherits = "{names[(index + 1) % length]}"')
    lines.append('[profile.default]\nglobal-timeout = "80m"')

    with pytest.raises(ProfileInheritanceError):
        resolved_global_timeout("\n".join(lines), names[0])