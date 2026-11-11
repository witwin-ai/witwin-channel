# Copyright Xingyu Chen.
# Run every survival case in a fresh process and require a clean exit with a usable CUDA context.

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tests.survival.cases import CASES, HEALTH_OK

ROOT = Path(__file__).resolve().parents[2]
SECOND_GPU_ENV = "CHANNEL_SURVIVAL_SECOND_GPU"
# Cases that keep the GPU busy for tens of seconds (a hang reproduction) run only on request,
# because a shared GPU's other tenants time out behind them.
LONG_WORK_ENV = "CHANNEL_SURVIVAL_LONG_GPU_WORK"

# Cases that crash, hang or accept invalid input today, keyed by case name, with the finding ID.
# They are strict expected failures: a fix makes them pass, which fails the suite until the
# entry is removed here.
EXPECTED_FAILURES: dict[str, str] = {}


def _environment(gpus: int) -> dict[str, str] | None:
    env = dict(os.environ)
    if gpus == 1:
        return env
    second = os.environ.get(SECOND_GPU_ENV)
    if second is None:
        return None
    first = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env["CUDA_VISIBLE_DEVICES"] = f"{first},{second}"
    return env


def _params() -> list[object]:
    params = []
    for name in CASES:
        marks = []
        if name in EXPECTED_FAILURES:
            marks.append(pytest.mark.xfail(strict=True, reason=EXPECTED_FAILURES[name]))
        params.append(pytest.param(name, marks=marks, id=name))
    return params


def test_expected_failures_name_real_cases() -> None:
    assert set(EXPECTED_FAILURES) <= set(CASES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="survival cases need CUDA")
@pytest.mark.parametrize("name", _params())
def test_case_leaves_cuda_context_usable(name: str) -> None:
    item = CASES[name]
    if item.long_gpu_work and os.environ.get(LONG_WORK_ENV) != "1":
        pytest.skip(f"long GPU work; set {LONG_WORK_ENV}=1 on a GPU nobody else uses")
    env = _environment(item.gpus)
    if env is None:
        pytest.skip(f"needs a second GPU; set {SECOND_GPU_ENV} to its physical index")
    try:
        completed = subprocess.run([sys.executable, "-m", "tests.survival.cases", name], cwd=ROOT,
                                   env=env, capture_output=True, text=True, timeout=item.timeout_s,
                                   check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{name} ({item.finding}) did not finish within {item.timeout_s:.0f} s")
    report = "\n".join(line for line in completed.stdout.splitlines() if line.startswith("SURVIVAL"))
    tail = "\n".join(completed.stderr.strip().splitlines()[-5:])
    assert completed.returncode == 0 and HEALTH_OK in completed.stdout, (
        f"{name} ({item.finding}) exit {completed.returncode}\n{report}\n{tail}")
