"""Smoke test: version string is importable and well-formed."""

import re

import aa_animator_v2
from aa_animator_v2._version import __version__


def test_version_is_string() -> None:
    assert isinstance(__version__, str)


def test_version_semver() -> None:
    # Accept N.N.N or N.N.N.devN etc.
    assert re.match(r"^\d+\.\d+\.\d+", __version__), f"bad version: {__version__!r}"


def test_package_version_matches_module() -> None:
    assert aa_animator_v2.__version__ == __version__
