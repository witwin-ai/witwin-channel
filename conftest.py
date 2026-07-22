"""Root conftest: configure the validated native extension for the test suite.

When the packaged extension is absent, search ``artifacts/cmake-*`` for a pyd
and build-fingerprint sidecar matching the running interpreter. Configure the
explicit developer loader variables and abort collection when no validated
build exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.support.native_ext import BUILD_GUIDANCE, inject_native_paths  # noqa: E402

if not inject_native_paths():
    pytest.exit(BUILD_GUIDANCE, returncode=4)
