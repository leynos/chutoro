"""Resolving a nextest profile's whole-run budget through its chain.

nextest resolves a profile's settings by walking each ``inherits`` link
from the selected profile to ``default``. The budget a lane is compared
against is the one that walk reaches, not the selected profile's own
table. This reading lives beside ``timeout_budgets`` rather than inside
it because that module already carries the per-test and job readings and
must stay under the 400-line limit ``AGENTS.md`` sets; it shares
``parse_config`` rather than parsing TOML a second time.

See "Test timeouts: four tiers, outermost last" in
``docs/developers-guide.md``.
"""

import typing as typ

from nextest_durations import _seconds
from timeout_budgets import parse_config


class ProfileInheritanceError(ValueError):
    """Raised when a profile resolves no whole-run budget.

    A cycle, an undeclared profile, or a chain that reaches ``default``
    without a ``global-timeout`` would otherwise read as an unbounded
    run, and every ordering comparison above it would pass against a
    budget nextest never applies.
    """


def _declared_profile(
    profiles: dict[str, typ.Any], name: str
) -> dict[str, typ.Any]:
    """Return a profile's table, refusing a name the configuration omits.

    Returns
    -------
    dict[str, typing.Any]
        The profile's own table.

    Raises
    ------
    ProfileInheritanceError
        If the name is not declared as a mapping.
    """
    section = profiles.get(name)
    if not isinstance(section, dict):
        message = f"nextest profile {name!r} is not declared"
        raise ProfileInheritanceError(message)
    return section


def _parent_profile(section: dict[str, typ.Any], current: str) -> str | None:
    """Return the next profile in the chain, or None at its root.

    A declared ``inherits`` link is returned as written, including a
    self-reference: the pinned nextest rejects ``ci.inherits = "ci"``, so
    the walk's visited-name guard must see it and refuse the cycle rather
    than substituting ``default``. A custom profile that names no parent
    still inherits ``default``, which is the root of every nextest chain,
    and ``default`` itself has no parent.

    Returns
    -------
    str or None
        The parent's name, or None when the walk has reached its root.
    """
    inherited = section.get("inherits")
    if isinstance(inherited, str):
        return inherited
    if current == "default":
        return None
    return "default"


def _inheritance_chain(
    profiles: dict[str, typ.Any], profile: str
) -> list[str]:
    """Return the profile names nextest consults, nearest first.

    Parameters
    ----------
    profiles : dict[str, typing.Any]
        The parsed ``profile`` table.
    profile : str
        The selected profile's name.

    Returns
    -------
    list[str]
        The selected profile followed by each ancestor, ending at
        ``default``.

    Raises
    ------
    ProfileInheritanceError
        If the chain names an undeclared profile or forms a cycle.
    """
    chain: list[str] = []
    current: str | None = profile
    while current is not None:
        if current in chain:
            message = (
                f"nextest profile inheritance forms a cycle: "
                f"{' -> '.join([*chain, current])}"
            )
            raise ProfileInheritanceError(message)
        chain.append(current)
        current = _parent_profile(_declared_profile(profiles, current), current)
    return chain


def resolved_global_timeout(config_text: str, profile: str = "default") -> float:
    """Return the ``global-timeout`` a profile resolves to, in seconds.

    nextest resolves a profile's settings by walking its ``inherits``
    chain, which always terminates at ``default``; a custom profile that
    names no parent inherits ``default`` directly. Reading only the
    selected profile's own table would report the heavy lane as having no
    whole-run budget when it inherits the CI profile's, and every
    per-test comparison against that budget would then be skipped or
    made against nothing.

    Parameters
    ----------
    config_text : str
        A nextest configuration file's text.
    profile : str
        The profile whose resolved budget is wanted.

    Returns
    -------
    float
        The whole-run budget in seconds.

    Raises
    ------
    ProfileInheritanceError
        If the chain names an undeclared profile, forms a cycle, or
        reaches ``default`` without a ``global-timeout``.
    """
    profiles = parse_config(config_text).get("profile")
    if not isinstance(profiles, dict):
        message = "nextest.toml must declare a [profile] table"
        raise ProfileInheritanceError(message)
    for name in _inheritance_chain(profiles, profile):
        budget = profiles[name].get("global-timeout")
        if isinstance(budget, str):
            return _seconds(budget)
    message = (
        f"nextest profile {profile!r} resolves no global-timeout; "
        f"[profile.default] must set one or the run is unbounded"
    )
    raise ProfileInheritanceError(message)