"""Fixtures shared by the workflow-contract modules.

Pytest makes a fixture defined in one test module invisible to another, so the
two Kani tier contracts would otherwise each have to read the same Makefile
targets. These fixtures read them once per module and hand the same text to
both: what a tier's commands say, and what happens when they run.

Re-use policy: import no module here; a fixture is requested by name.
"""

from __future__ import annotations

import pytest

from kani_recipe_reading import (
    KANI_FULL_TARGET_RE,
    KANI_TARGET_RE,
    MAKEFILE_PATH,
    recipe,
)


@pytest.fixture(scope="module")
def makefile() -> str:
    """Return the Makefile's text."""
    return MAKEFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fast_recipe(makefile: str) -> str:
    """Return the recipe body of the ``kani:`` target."""
    return recipe(makefile, KANI_TARGET_RE, "kani")


@pytest.fixture(scope="module")
def full_recipe(makefile: str) -> str:
    """Return the recipe body of the ``kani-full:`` target."""
    return recipe(makefile, KANI_FULL_TARGET_RE, "kani-full")
