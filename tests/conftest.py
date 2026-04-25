"""
conftest.py — make the src layout importable for all tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
