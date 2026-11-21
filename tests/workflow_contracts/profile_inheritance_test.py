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
from profile_inheritance import ProfileInheritanceError, resolved_global_timeout


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