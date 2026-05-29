"""
Integration tests require the free-threaded CPython 3.14+ build.
Skip automatically on older runtimes.
"""

import sys

import pytest

if sys.version_info < (3, 14):
    collect_ignore_glob = ["*.py"]

    @pytest.fixture(autouse=True)
    def _require_freethreaded() -> None:
        pytest.skip("Integration tests require free-threaded CPython 3.14+")
