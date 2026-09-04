"""Contract-test who owns each cached path.

A mutable cache path with two owners has two keys racing to describe it, so
one of them always restores work the other has already invalidated. Every
path below therefore has exactly one owner per job, and every key is
derived from a correctness input a reader can explain. Compiler output is
excluded outright: sccache owns it, and archiving a ``target`` tree beside
sccache duplicates that ownership and inflates the weekly quota.

Run via ``make test-workflow-contracts``.
"""

from __future__ import annotations

import collections
import collections.abc as cabc
import re
from pathlib import Path

import pytest
from workflow_support import (
    CACHE_ACTION_SHA,
    SHARED_ACTION_OWNED_PATHS,
    cache_is_enabled,
    declared_cache_paths,
    jobs,
    load_workflow,
    shared_action_name,
    run_script,
    steps,
    uses_reference,
    workflow_names,
    workflow_paths,
)

CACHE_ACTION_RE = re.compile(r"^actions/cache(?:/(?:restore|save))?@(?P<sha>\S+)$")

#: setup-uv owns ~/.cache/uv whenever its cache is enabled, wherever it is
#: invoked from.
UV_CACHE_PATH = "~/.cache/uv"


def _cache_steps(definition: dict) -> list[dict]:
    """Return the job's explicit cache steps."""
    return [
        step
        for step in steps(definition)
        if CACHE_ACTION_RE.match(uses_reference(step))
    ]


def _explicit_owner(step: dict, reference: str) -> list[tuple[str, str]]:
    """Return the paths a cache step names outright."""
    label = step.get("name", reference)
    return [(path, label) for path in declared_cache_paths(step)]


def _implicit_owner(step: dict, reference: str) -> list[tuple[str, str]]:
    """Return the paths an action caches without the caller naming them."""
    if not cache_is_enabled(step):
        return []
    if reference.startswith("astral-sh/setup-uv@"):
        return [(UV_CACHE_PATH, "astral-sh/setup-uv")]
    action = shared_action_name(reference)
    if action in SHARED_ACTION_OWNED_PATHS:
        return [(path, action) for path in SHARED_ACTION_OWNED_PATHS[action]]
    return []


def _step_owners(step: dict) -> list[tuple[str, str]]:
    """Return every (path, owner) pair one step establishes."""
    reference = uses_reference(step)
    if CACHE_ACTION_RE.match(reference):
        return _explicit_owner(step, reference)
    return _implicit_owner(step, reference)


def _owned_paths(definition: dict) -> list[tuple[str, str]]:
    """Return every (path, owner) pair a job establishes."""
    return [pair for step in steps(definition) for pair in _step_owners(step)]


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_every_cache_step_uses_the_pinned_cache_action(workflow_name: str) -> None:
    """One reviewed cache implementation, everywhere."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            match = CACHE_ACTION_RE.match(uses_reference(step))
            assert match is not None
            assert match.group("sha") == CACHE_ACTION_SHA, (
                f"{workflow_name}:{job_name} pins the cache action at "
                f"{match.group('sha')}, not {CACHE_ACTION_SHA}"
            )


@pytest.mark.parametrize("workflow_path", workflow_paths(), ids=workflow_names())
def test_no_workflow_uses_the_ubicloud_cache_fork(workflow_path: Path) -> None:
    """The transparent proxy intercepts actions/cache, so the fork is dead."""
    assert "ubicloud/cache" not in workflow_path.read_text(encoding="utf-8"), (
        f"{workflow_path.name} must use actions/cache, not the deprecated "
        "ubicloud/cache fork"
    )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_no_cache_step_archives_a_target_tree(workflow_name: str) -> None:
    """sccache is the sole owner of compiler output."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            for path in declared_cache_paths(step):
                assert not re.search(r"(^|/)target(/|$)", path), (
                    f"{workflow_name}:{job_name} archives {path}; sccache owns "
                    "compiler output and target trees must not be cached"
                )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_each_cached_path_has_exactly_one_owner(workflow_name: str) -> None:
    """Two owners for one path means two keys racing to describe it."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        counts = collections.Counter(path for path, _ in _owned_paths(definition))
        duplicated = {path: count for path, count in counts.items() if count > 1}
        assert not duplicated, (
            f"{workflow_name}:{job_name} gives these paths more than one cache "
            f"owner: {sorted(duplicated)}"
        )


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_every_cache_step_declares_an_explainable_key(workflow_name: str) -> None:
    """A cache without a key cannot explain a miss."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            with_block = step.get("with", {})
            key = with_block.get("key")
            assert isinstance(key, str) and key.strip(), (
                f"{workflow_name}:{job_name} declares a cache step without a key"
            )


#: Paths that must never reappear in a cache step. Kani's three parts total
#: about 1.8 GB against a 16 s cold install (run 33819842254), so caching them
#: costs far more runner time than it saves. Removing the old contract without
#: this one would let the archive come back unnoticed.
REJECTED_CACHE_PATHS: tuple[str, ...] = (
    "~/.cargo/bin/cargo-kani",
    "~/.cargo/bin/kani",
    "~/.kani",
    "~/.kani-rustup",
)


def _normalize(path: str) -> str:
    """Strip a trailing separator so equivalent spellings compare equal."""
    return path.rstrip("/") or "/"


#: Characters that make an `actions/cache` path a pattern rather than a
#: literal. The action resolves its paths through `@actions/glob`, so a
#: contract that compares them as plain strings can be walked straight past:
#: `~/.cargo/bin/*` archives `cargo-kani` while never spelling its name.
GLOB_METACHARACTERS = "*?["


def _has_glob(pattern: str) -> bool:
    """Report whether a declared path is a glob pattern."""
    return any(character in pattern for character in GLOB_METACHARACTERS)


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate an `@actions/glob` pattern into an equivalent regex.

    `**` crosses separators; `*` and `?` do not. A bracket expression is
    copied through as a character class, with `!` rewritten to the regex
    spelling of negation.
    """
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        elif pattern[index] == "[":
            close = pattern.find("]", index + 1)
            if close == -1:
                parts.append(re.escape("["))
                index += 1
                continue
            body = pattern[index + 1 : close]
            if body.startswith("!"):
                body = "^" + body[1:]
            parts.append(f"[{body}]")
            index = close + 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts) + r"\Z")


def _literal_prefix(pattern: str) -> str:
    """Return the leading segments of a pattern that contain no wildcard.

    Everything a pattern can match lives under this directory, which is what
    makes it the right thing to compare against when asking whether the
    archive would reach inside a rejected tree.
    """
    kept: list[str] = []
    for segment in pattern.split("/"):
        if _has_glob(segment):
            break
        kept.append(segment)
    return "/".join(kept) or "/"


def _ancestors(path: str) -> list[str]:
    """Return every proper ancestor directory of a path, shallowest first."""
    segments = path.split("/")
    return ["/".join(segments[:count]) for count in range(1, len(segments))]


def _covers(declared: str, rejected: str) -> bool:
    """Report whether a declared cache path would archive a rejected one.

    Exact equality is not enough, in three separate ways. `actions/cache`
    accepts a directory with or without a trailing separator. Caching a
    parent sweeps its children in, so `~/.cargo/bin` archives `cargo-kani`
    just as surely as naming it; `@actions/glob` sets `implicitDescendants`,
    so a pattern matching that parent does the same. And a child counts too:
    `~/.kani-rustup/toolchains` is most of the 1.3 GB.
    """
    left, right = _normalize(declared), _normalize(rejected)
    matches = _glob_to_regex(left).fullmatch
    if matches(right):
        return True
    if any(matches(ancestor) for ancestor in _ancestors(right)):
        return True
    prefix = _normalize(_literal_prefix(left))
    return prefix == right or prefix.startswith(f"{right}/")


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_the_rejected_caches_stay_rejected(workflow_name: str) -> None:
    """A cache that failed the payoff rule must not quietly return."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        for step in _cache_steps(definition):
            for path in declared_cache_paths(step):
                covered = [
                    rejected
                    for rejected in REJECTED_CACHE_PATHS
                    if _covers(path, rejected)
                ]
                assert not covered, (
                    f"{workflow_name}:{job_name} caches {path}, which covers "
                    f"{covered}; those were measured and rejected. See the "
                    "developers guide."
                )


@pytest.mark.parametrize(
    ("declared", "is_covered"),
    [
        pytest.param("~/.kani", True, id="exact"),
        pytest.param("~/.kani/", True, id="trailing-separator"),
        pytest.param("~/.cargo/bin", True, id="parent-directory"),
        pytest.param("~/.kani-rustup/toolchains", True, id="child-directory"),
        pytest.param("~/.cargo/registry", False, id="unrelated-sibling"),
        pytest.param("~/.cargo/bin/whitaker-installer", False, id="other-tool"),
        pytest.param("~/.kani-extra", False, id="shared-prefix-only"),
        pytest.param("~/.cargo/bin/*", True, id="globbed-parent"),
        pytest.param("~/.cargo/**", True, id="recursive-glob"),
        pytest.param("~/.kani*", True, id="glob-matching-the-path-itself"),
        pytest.param("~/.kani-rustup/*/lib", True, id="glob-inside-a-rejected-tree"),
        pytest.param("~/.cargo/bin/whitaker-*", False, id="glob-for-another-tool"),
        pytest.param("~/.cargo/registry/**", False, id="recursive-glob-elsewhere"),
        pytest.param("~/.cargo/bin/?ani", True, id="single-character-wildcard"),
        pytest.param("~/.cargo/bin/[ck]ani", True, id="bracket-expression"),
    ],
)
def test_rejected_path_coverage_recognizes_equivalent_spellings(
    declared: str, is_covered: bool
) -> None:
    """Guard the guard, since exact equality was the original hole."""
    covered = any(_covers(declared, rejected) for rejected in REJECTED_CACHE_PATHS)
    assert covered is is_covered, (
        f"{declared} should {'' if is_covered else 'not '}count as covering a "
        "rejected cache path"
    )


#: Each installer script and the cache-step path that must restore its work
#: first. Pairing them by tool keeps the ordering check meaningful once a job
#: installs more than one tool.
INSTALLER_CACHE_PAIRS = (("scripts/install-verus.sh", ".verus"),)


def _first_index(items: list[str], predicate: cabc.Callable[[str], bool]) -> int | None:
    """Return the first index whose item satisfies the predicate."""
    return next((index for index, item in enumerate(items) if predicate(item)), None)


@pytest.mark.parametrize("workflow_name", workflow_names())
def test_cache_setup_precedes_the_install_it_protects(workflow_name: str) -> None:
    """A cache restored after an install has already paid for the install."""
    for job_name, definition in jobs(load_workflow(workflow_name)).items():
        job_steps = steps(definition)
        scripts = [run_script(step) for step in job_steps]
        for installer, cached_path in INSTALLER_CACHE_PAIRS:
            install_at = _first_index(scripts, lambda s, i=installer: i in s)
            if install_at is None:
                continue
            cache_at = next(
                (
                    index
                    for index, step in enumerate(job_steps)
                    if CACHE_ACTION_RE.match(uses_reference(step))
                    and any(
                        path.endswith(cached_path)
                        for path in declared_cache_paths(step)
                    )
                ),
                None,
            )
            assert cache_at is not None, (
                f"{workflow_name}:{job_name} runs {installer} with no cache step "
                f"owning {cached_path}"
            )
            assert cache_at < install_at, (
                f"{workflow_name}:{job_name} restores the {cached_path} cache "
                "after the install it is meant to avoid"
            )
