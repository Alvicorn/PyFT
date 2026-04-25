"""
Integration tests require Python 3.12+ (sys.monitoring / SyncMonitor).
Skip automatically on older runtimes.
"""

import sys

import pytest

if sys.version_info < (3, 12):
    collect_ignore_glob = ["*.py"]

    @pytest.fixture(autouse=True)
    def _require_monitoring():
        pytest.skip("Integration tests require Python 3.12+ (sys.monitoring)")
