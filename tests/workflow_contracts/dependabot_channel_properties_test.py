"""Exercise the pinned Rust channel contract over generated inputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given
from hypothesis import strategies as st

from dependabot_config_test import RepositoryConfigReadError, read_rust_toolchain_manifest


_VERSION_COMPONENT = st.integers(min_value=0, max_value=9999)
_DATE = st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31))


def _write_channel(path: Path, channel: str) -> None:
    path.write_text(f'[toolchain]\nchannel = "{channel}"\n', encoding="utf-8")


@st.composite
def _pinned_channels(draw: st.DrawFn) -> str:
    """Generate exact versions and dated nightly or beta channels."""
    kind = draw(st.sampled_from(("version", "nightly", "beta")))
    if kind == "version":
        components = draw(
            st.lists(_VERSION_COMPONENT, min_size=3, max_size=3)
        )
        return ".".join(map(str, components))

    release_date = draw(_DATE)
    return f"{kind}-{release_date.isoformat()}"


@given(channel=_pinned_channels())
def test_generated_pinned_channels_are_accepted(channel: str) -> None:
    """Accept exact versions and every valid generated nightly or beta date."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rust-toolchain.toml"
        _write_channel(path, channel)
        assert read_rust_toolchain_manifest(path)["toolchain"]["channel"] == channel


@st.composite
def _invalid_channels(draw: st.DrawFn) -> str:
    """Generate floating, malformed, near-prefix and impossible-date channels."""
    kind = draw(st.sampled_from(("floating", "components", "date", "near-miss")))
    if kind == "floating":
        return draw(st.sampled_from(("stable", "beta", "nightly")))
    if kind == "components":
        components = draw(
            st.one_of(
                st.lists(_VERSION_COMPONENT, min_size=1, max_size=2),
                st.lists(_VERSION_COMPONENT, min_size=4, max_size=4),
            )
        )
        return ".".join(map(str, components))

    release_date = draw(_DATE)
    if kind == "date":
        return f"nightly-{release_date.year:04d}-{release_date.month:02d}-32"
    prefix, separator = draw(
        st.sampled_from(
            (
                ("nightlies", "-"),
                ("nightlyx", "-"),
                ("betas", "-"),
                ("betax", "-"),
                ("nightly", "_"),
                ("beta", "/"),
                ("nightly", "."),
            )
        )
    )
    return f"{prefix}{separator}{release_date.isoformat()}"


@given(channel=_invalid_channels())
def test_generated_unpinned_and_malformed_channels_are_rejected(channel: str) -> None:
    """Reject generated floating, malformed and near-miss toolchain channels."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rust-toolchain.toml"
        _write_channel(path, channel)
        try:
            read_rust_toolchain_manifest(path)
        except RepositoryConfigReadError:
            pass
        else:
            raise AssertionError(f"invalid toolchain channel was accepted: {channel!r}")
