"""CI budget gates for the AD suite (plan 07 section 10).

Two families of budgets, both of which fail the suite on regression instead
of drifting silently:

- The gradient-error tolerances and FD steps of tests/ad/_tolerances.py are
  pinned to their agreed values. Loosening a tolerance to make a failing
  gradient pass is the failure mode this gate exists to catch; tightening is
  fine but must be done here and in _tolerances.py together.
- The AD execution overhead relative to the primal solve is bounded: forward
  time in the AD modes, the wall time of one reverse pass, the retained tape
  and the CUDA peak-memory high-water mark. The budgets carry generous CI
  headroom (measured medians sit several times below them); they are meant
  to catch order-of-magnitude regressions (an accidental sync storm, a
  Python re-solve on the hot path, tape blow-up), not scheduler noise.
"""

from __future__ import annotations

from statistics import median
from time import perf_counter

import pytest
import torch

from tests.ad import _tolerances
from tests.support.scenes import same_side_wall_reflection_scene
from witwin.channel_native.deterministic import Config as DeterministicConfig
from witwin.channel_native.deterministic import solve as deterministic_solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for AD budgets"
)


def test_gradient_tolerances_are_pinned():
    """Section 9.2 constants, frozen. Change them HERE and in _tolerances.py."""

    assert _tolerances.REL_TOL_PATH == 5.0e-3
    assert _tolerances.REL_TOL_GENERAL == 5.0e-2
    assert _tolerances.ABS_TOL == 1.0e-12
    assert _tolerances.FD_STEP_POSITION == 1.0e-2
    assert _tolerances.FD_STEP_POSITION_PHASE == 1.0e-3
    assert _tolerances.FD_STEP_VERTEX == 1.0e-3
    assert _tolerances.FD_STEP_GEOMETRY == 1.0e-3
    assert _tolerances.FD_STEP_EPS_R == 1.0e-2
    assert _tolerances.FD_STEP_SIGMA_E == 1.0e-3
    assert _tolerances.FD_STEP_GAIN == 1.0e-3
    assert _tolerances.FD_STEP_THICKNESS == 1.0e-4
    assert _tolerances.FD_REL_STEP_FREQUENCY == 1.0e-4


_REPEATS = 5

# Execution budgets (relative to the primal solve of the same scene, plus an
# absolute floor so millisecond-scale solves are not judged by timer noise).
_AD_FORWARD_OVERHEAD_MAX = 6.0
_AD_FORWARD_FLOOR_MS = 60.0
_BACKWARD_OVERHEAD_MAX = 6.0
_BACKWARD_FLOOR_MS = 120.0
_PEAK_MEMORY_OVERHEAD_MAX = 4.0
_PEAK_MEMORY_FLOOR_BYTES = 64 * 2**20
_TAPE_BYTES_MAX = 1 * 2**20  # single-wall scene: a handful of small rows


def _config(ad_mode: str) -> DeterministicConfig:
    return DeterministicConfig(
        max_depth=1,
        components=frozenset({"los", "reflection"}),
        export_paths=True,
        ad_mode=ad_mode,
    )


def _timed_solve(scene, ad_mode: str):
    torch.cuda.synchronize()
    start = perf_counter()
    result = deterministic_solve(scene, _config(ad_mode))
    torch.cuda.synchronize()
    return result, (perf_counter() - start) * 1.0e3


def test_ad_time_memory_and_tape_budgets():
    scene = same_side_wall_reflection_scene()
    leaf = scene.compile().materials.eps_r
    # Warm up caches and the CUDA context before measuring.
    _timed_solve(scene, "none")

    primal_ms = []
    for _ in range(_REPEATS):
        _, elapsed = _timed_solve(scene, "none")
        primal_ms.append(elapsed)
    primal_forward = median(primal_ms)

    leaf.requires_grad_(True)
    try:
        forward_ms = []
        backward_ms = []
        tape_bytes = 0
        for _ in range(_REPEATS):
            result, elapsed = _timed_solve(scene, "vjp")
            forward_ms.append(elapsed)
            tape_bytes = int(result.metadata["kernel"]["tape_bytes"])
            loss = result.paths.coefficient.real.sum()
            torch.cuda.synchronize()
            start = perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_ms.append((perf_counter() - start) * 1.0e3)
            leaf.grad = None

        ad_forward = median(forward_ms)
        backward = median(backward_ms)
        assert ad_forward <= max(
            _AD_FORWARD_OVERHEAD_MAX * primal_forward, _AD_FORWARD_FLOOR_MS
        ), f"AD forward regressed: {ad_forward:.2f}ms vs primal {primal_forward:.2f}ms"
        assert backward <= max(
            _BACKWARD_OVERHEAD_MAX * primal_forward, _BACKWARD_FLOOR_MS
        ), f"reverse pass regressed: {backward:.2f}ms vs primal {primal_forward:.2f}ms"
        assert 0 < tape_bytes <= _TAPE_BYTES_MAX, (
            f"vjp tape for the single-wall scene is {tape_bytes} bytes"
        )

        # Peak-memory high-water mark of one AD solve vs one primal solve.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        deterministic_solve(scene, _config("none"))
        torch.cuda.synchronize()
        primal_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = deterministic_solve(scene, _config("vjp"))
        result.paths.coefficient.real.sum().backward()
        torch.cuda.synchronize()
        ad_peak = torch.cuda.max_memory_allocated()
        leaf.grad = None
        assert ad_peak <= max(
            _PEAK_MEMORY_OVERHEAD_MAX * primal_peak, _PEAK_MEMORY_FLOOR_BYTES
        ), f"AD peak memory regressed: {ad_peak} vs primal {primal_peak}"
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


def test_metadata_reports_solve_timing():
    scene = same_side_wall_reflection_scene()
    result = deterministic_solve(scene, _config("none"))
    kernel = result.metadata["kernel"]
    assert kernel["forward_time_ms"] > 0.0
    assert kernel["peak_memory_bytes"] >= 0
