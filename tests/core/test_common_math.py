from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from witwin.core import ReceiverGrid
from witwin.core import normalize_vec3
from tests.support.core_world import make_receiver_grid
from witwin.channel.components import (
    DEFAULT_COMPONENTS,
    component_availability_status,
    validated_components,
)
from witwin.channel.scene.endpoints import (
    axis_aligned_grid_spec,
    component_grid_shape,
    first_receiver_grid,
)
from witwin.channel.scene.endpoints import (
    ReceiverGrid as RuntimeReceiverGrid,
)


def _grid() -> ReceiverGrid:
    return make_receiver_grid(
        origin=torch.tensor([2.0, -1.0, 3.0]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(3, 5),
        spacing=(0.5, 0.25),
    )


def test_axis_aligned_grid_spec_preserves_native_layout_contract():
    grid = _grid()
    spec = axis_aligned_grid_spec(grid)

    assert spec.grid is grid
    assert spec.axis == 0
    assert spec.position == 2.0
    assert spec.resolution0 == 3
    assert spec.resolution1 == 5
    assert spec.coord0_min == pytest.approx(-1.25)
    assert spec.coord0_max == pytest.approx(0.25)
    assert spec.coord1_min == pytest.approx(2.875)
    assert spec.coord1_max == pytest.approx(4.125)
    assert spec.cell_area == pytest.approx(0.125)
    assert component_grid_shape(grid) == (5, 3)


def test_first_receiver_grid_and_component_validation_are_shared():
    grid = _grid()
    runtime_grid = RuntimeReceiverGrid(source=grid)
    scene = SimpleNamespace(receivers=[runtime_grid])

    assert first_receiver_grid(scene) is runtime_grid
    assert runtime_grid.source is grid
    assert validated_components(
        DEFAULT_COMPONENTS, error_message="bad {valid}"
    ) == DEFAULT_COMPONENTS
    with pytest.raises(ValueError, match="bad"):
        validated_components({"unknown"}, error_message="bad {valid}")


def test_component_status_keeps_solver_specific_errors():
    with pytest.raises(RuntimeError, match="reflection unavailable"):
        component_availability_status(
            {"reflection"},
            reflection_available=False,
            diffraction_available=True,
            reflection_error="reflection unavailable",
            diffraction_error="diffraction unavailable",
        )


def test_normalize_vec3_matches_frozen_expression():
    values = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    expected = values / torch.linalg.vector_norm(
        values, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)

    torch.testing.assert_close(normalize_vec3(values), expected, rtol=0.0, atol=0.0)


def test_core_modules_do_not_depend_on_solver_packages():
    core_dir = Path(__file__).resolve().parents[2] / "witwin" / "channel" / "core"
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        assert not any(
            module.startswith("witwin.channel.montecarlo")
            or module.startswith("witwin.channel.deterministic")
            for module in imported
        ), path.name
