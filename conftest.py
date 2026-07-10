"""Root conftest: make the native extensions importable for the test suite.

If ``_channel_native``/``_raydn`` are not already importable (e.g. via an
externally set PYTHONPATH), search ``artifacts/cmake-*`` build trees for pyds
matching the running interpreter. Abort collection with actionable guidance
when no usable build exists instead of failing every test with an opaque
ImportError.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.support.native_ext import BUILD_GUIDANCE, inject_native_paths

if not inject_native_paths():
    pytest.exit(BUILD_GUIDANCE, returncode=4)
