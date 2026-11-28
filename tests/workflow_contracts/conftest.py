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
    KANI_MAKEFILE_PATH,
    recipe,
)
from kani_reader_support import _write_package


@pytest.fixture(scope="module")
def makefile() -> str:
    """Return the Kani fragment's text."""
    return KANI_MAKEFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fast_recipe(makefile: str) -> str:
    """Return the recipe body of the ``kani:`` target."""
    return recipe(makefile, KANI_TARGET_RE, "kani")


@pytest.fixture(scope="module")
def full_recipe(makefile: str) -> str:
    """Return the recipe body of the ``kani-full:`` target."""
    return recipe(makefile, KANI_FULL_TARGET_RE, "kani-full")


@pytest.fixture
def write_package(tmp_path):
    """Return a function that writes one package under this test's tmp_path.

    The reader contracts build a package for nearly every case, so binding the
    root here keeps the source text the only argument each test supplies. A
    case that needs a custom layout still calls the shared writer with an
    explicit root.
    """

    def write(source: str, name: str = "fixture"):
        """Write one package with `source` as its library root."""
        return _write_package(tmp_path, source, name)

    return write
