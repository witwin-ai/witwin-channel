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

# Cases that crash, hang or accept invalid input today, keyed by case name, with the finding ID.
# They are strict expected failures: a fix makes them pass, which fails the suite until the
# entry is removed here.
EXPECTED_FAILURES = {
    "s1.box_transmission_depth1.det": "S-1 device trap when hits exceed max_depth",
    "s1.box_transmission_depth1.path": "S-1 device trap when hits exceed max_depth",
    "s1.box_transmission_depth1.bdpt": "S-1 device trap when hits exceed max_depth",
    "s1.city4_transmission_depth2.mc": "S-1 device trap when hits exceed max_depth",
    "s1.city4_transmission_depth2.det": "S-1 device trap when hits exceed max_depth",
    "s1.city8_transmission_depth5.det": "S-1 device trap at the maximum depth",
    "s1.consumer_two_walls_transmission_depth1": "S-1 device trap through the consumer",
    "s2.path_one_building_4rx_coupled": "S-2 coupled D-D stationary-root device assert",
    "s2.det_city4_grid16_coupled": "S-2 coupled D-D stationary-root device assert",
    "s3.nan_rx_transmission.det": "S-3 non-finite endpoint reaches the transmission trap",
    "s3.inf_tx_transmission.det": "S-3 non-finite endpoint reaches the transmission trap",
    "s3.far_rx_1e30_transmission.det": "S-3 far endpoint reaches the transmission trap",
    "s3.consumer_nan_source_transmission": "S-3 non-finite consumer endpoint",
    "s4.face_index_int64_wraps_to_valid": "S-4 int64 face index silently truncated to int32",
    "s5.consumer_endpoints_on_other_gpu": "S-5 consumer preflight does not compare devices",
    "s6.reevaluate_primitive_sequence_oob": "S-6 tampered topology index out of bounds",
    "s6.reevaluate_provenance_none_other_scene": "S-6 hand-built topology on another scene",
    "s8.kirchhoff_300ghz": "S-8 Kirchhoff series unbounded in k0*sigma_h (hang)",
    "s8.kirchhoff_1thz": "S-8 Kirchhoff series unbounded in k0*sigma_h (hang)",
}


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
