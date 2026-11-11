# Copyright Xingyu Chen.
# Runs every script under examples/ with small arguments so the examples keep working.

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CASES = {
    "radiomap_from_obj.py": ["--cells", "32", "--output", "{tmp}/map.png"],
    "radiomap_from_obj.py --demo-xml": ["--demo-xml", "--cells", "32", "--output", "{tmp}/x.png"],
    "path_cir_cfr.py": [],
    "material_calibration.py": [],
    "moving_tx_time_varying.py": ["--slots", "8", "--rediscover-every", "4"],
}


def test_every_example_has_a_case() -> None:
    scripts = {path.name for path in EXAMPLES.glob("*.py") if path.name != "street_canyon.py"}
    assert scripts == {case.split()[0] for case in CASES}


@pytest.mark.parametrize("case", sorted(CASES))
def test_example_runs(case: str, tmp_path: Path) -> None:
    script = EXAMPLES / case.split()[0]
    args = [arg.format(tmp=tmp_path) for arg in CASES[case]]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}

    completed = subprocess.run(
        [sys.executable, str(script), *args], cwd=tmp_path, env=env, capture_output=True,
        text=True, timeout=600, check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    if case == "material_calibration.py":
        assert "|error| 0.00" in completed.stdout, completed.stdout
