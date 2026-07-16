from __future__ import annotations

import pytest

from benchmarks.bench_solver_scaling import _expanded_scene, _operation


@pytest.mark.parametrize("solver", ("basic", "bdpt"))
def test_scaling_mc_operations_use_the_explicit_workspace_budget(solver: str):
    budget = 16 * (1 << 30)
    operation = _operation(
        solver,
        _expanded_scene(1, 1),
        depth=1,
        samples=1_000_000,
        workspace_limit_bytes=budget,
    )
    config = next(
        cell.cell_contents
        for cell in operation.__closure__ or ()
        if hasattr(cell.cell_contents, "workspace_limit_bytes")
    )

    assert config.workspace_limit_bytes == budget
