from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from witwin.channel_native.core import path_topology as legacy
from witwin.channel_native.core import scene_tensors
from witwin.channel_native.deterministic import solver


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_frequency_scalar_preserves_detached_scalar_contract():
    assert scene_tensors._frequency_scalar(SimpleNamespace(frequency=2.4)) == 2.4

    frequency = torch.tensor(2.4, dtype=torch.float64, requires_grad=True)
    assert scene_tensors._frequency_scalar(SimpleNamespace(frequency=frequency)) == 2.4


def test_frequency_scalar_is_same_object_compatibility_export():
    owner = scene_tensors._frequency_scalar

    assert owner.__module__ == scene_tensors.__name__
    assert legacy._frequency_scalar is owner
    assert solver._frequency_scalar is owner


@pytest.mark.parametrize(
    "imports",
    (
        (
            "from witwin.channel_native.core import scene_tensors; "
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.deterministic import solver"
        ),
        (
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.deterministic import solver; "
            "from witwin.channel_native.core import scene_tensors"
        ),
        (
            "from witwin.channel_native.deterministic import solver; "
            "from witwin.channel_native.core import path_topology as legacy; "
            "from witwin.channel_native.core import scene_tensors"
        ),
    ),
)
def test_frequency_scalar_import_order_preserves_facade_identity(imports: str):
    code = (
        f"{imports}; "
        "assert legacy._frequency_scalar is scene_tensors._frequency_scalar "
        "is solver._frequency_scalar"
    )
    environment = os.environ.copy()
    source_root = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
