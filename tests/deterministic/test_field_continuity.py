# Copyright Xingyu Chen.
# Test deterministic field continuity across visibility boundaries.

"""Test deterministic field continuity across visibility boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.support.core_world import (
    make_mesh_structure,
    make_receiver_grid,
    make_transmitter,
)
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config, solve
from witwin.core import PhysicalMaterial, Scene


_FREQUENCY_HZ = 5.0e9
_TX_POSITION = (-0.2, -0.5, 0.42)
_TX_POLARIZATION = (0.0, 0.0, 1.0)
_RX_POLARIZATION = (0.0, 0.0, 1.0)
_CUBE_CENTER = (0.0, 0.0, 0.15)
_CUBE_SIZE_M = 0.2
_PLANE_Z = 0.10
_FULL_COMPONENTS = frozenset({"los", "reflection", "diffraction", "transmission"})
_SUPPORT_FLOOR_DB = -80.0  # matches verify-e1e support_active gate


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the deterministic field continuity tests")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")


def _cube_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    cx, cy, cz = _CUBE_CENTER
    half = _CUBE_SIZE_M / 2.0
    vertices = torch.tensor(
        [
            [cx - half, cy - half, cz - half],
            [cx + half, cy - half, cz - half],
            [cx + half, cy + half, cz - half],
            [cx - half, cy + half, cz - half],
            [cx - half, cy - half, cz + half],
            [cx + half, cy - half, cz + half],
            [cx + half, cy + half, cz + half],
            [cx - half, cy + half, cz + half],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.int32,
    )
    return vertices, faces


def _build_scene(x_values: np.ndarray, y_values: np.ndarray) -> Scene:
    """Single-cube PEC scene with an explicit axis-aligned receiver grid.

 Mirrors ``benchmarks/fullwave_validation/scenarios.build_channel_scene`` and
 the verify-e1e ``_env.build_scene`` helper, but takes explicit 1-D receiver
 x/y arrays so each test can use a small grid.
 """
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    vertices, faces = _cube_mesh()
    structure = make_mesh_structure(
        vertices=vertices,
        faces=faces,
        material=PhysicalMaterial.perfect_conductor(name="pec"),
        name="cube-1",
        surface_id=1,
    )
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    grid = make_receiver_grid(
        origin=torch.tensor([float(x[0]), float(y[0]), _PLANE_Z]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(x.size, y.size),
        spacing=(dx, dy),
        polarization=torch.tensor(list(_RX_POLARIZATION)),
    )
    return Scene(
        structures=[structure],
        endpoints=[
            make_transmitter(
                position=torch.tensor(list(_TX_POSITION)),
                polarization=torch.tensor(list(_TX_POLARIZATION)),
            ),
            grid,
        ],
        metadata={"fullwave_validation_case": "single_cube-metal"},
    )


def _solve_maps(
    x_values: np.ndarray,
    y_values: np.ndarray,
    components: frozenset[str] | set[str],
    max_depth: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the (ny, nx) total field map and per-component z-projected maps.

 ``result.field`` and ``result.component_fields[*]`` are the coherent scalar
 already projected onto the receiver polarization (z), so with z-polarized
 receivers they are exactly the z-components of the total / per-component
 field vectors (the physical observable compared against Maxwell Ez).
 """
    result = solve(
        _build_scene(x_values, y_values),
        Config(
            components=components,
            max_depth=max_depth,
            max_diffraction_order=1,
            coherent=True,
            return_field=True,
            export_paths=True,
            diagnostics=True,
        ),
        reference_frequency_hz=_FREQUENCY_HZ,
    )
    field = result.field.detach().cpu().numpy()[0]
    component = {
        name: values.detach().cpu().numpy()[0]
        for name, values in result.component_fields.items()
        if values.numel() > 0
    }
    return field, component


def _footprint_valid(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """(ny, nx) mask that is False for receivers inside the cube footprint.

 The z=0.10 plane cuts the PEC cube (|x| <= 0.1, |y| <= 0.1); those receivers
 are inside the conductor and carry no field. Same predicate as verify-e1e.
 """
    x_inside = np.abs(np.asarray(x_values))[None, :] <= 0.1
    y_inside = np.abs(np.asarray(y_values))[:, None] <= 0.1
    return ~(x_inside & y_inside)


def _support_active(component: np.ndarray) -> np.ndarray:
    magnitude = np.abs(component)
    peak = float(magnitude.max())
    if peak <= 0.0:
        return np.zeros_like(magnitude, dtype=bool)
    return magnitude > peak * 10.0 ** (_SUPPORT_FLOOR_DB / 20.0)


def _db(magnitude: np.ndarray) -> np.ndarray:
    floor = max(float(magnitude.max()) * 1.0e-10, 1.0e-30)
    return 20.0 * np.log10(np.maximum(magnitude, floor))


def test_t1_isb_rsb_toggle_continuity() -> None:
    """Assert total-field continuity across LoS and reflection visibility boundaries."""
    _require_cuda()
    x = np.linspace(-0.4, 0.4, 64)
    y = np.linspace(-0.4, 0.4, 64)
    field, component = _solve_maps(x, y, _FULL_COMPONENTS, max_depth=2)
    valid = _footprint_valid(x, y)
    los_active = _support_active(component["los"])
    reflection_active = _support_active(component["reflection"])
    field_db = _db(np.abs(field))

    jumps: list[np.ndarray] = []
    for axis in ("x", "y"):
        if axis == "x":
            pair_valid = valid[:, 1:] & valid[:, :-1]
            toggled = (los_active[:, 1:] != los_active[:, :-1]) | (
                reflection_active[:, 1:] != reflection_active[:, :-1]
            )
            jump = np.abs(field_db[:, 1:] - field_db[:, :-1])
        else:
            pair_valid = valid[1:, :] & valid[:-1, :]
            toggled = (los_active[1:, :] != los_active[:-1, :]) | (
                reflection_active[1:, :] != reflection_active[:-1, :]
            )
            jump = np.abs(field_db[1:, :] - field_db[:-1, :])
        jumps.append(jump[pair_valid & toggled])
    toggle_jumps = np.concatenate(jumps)

    assert toggle_jumps.size > 0, "no ISB/RSB toggle pairs found on the grid"
    median_db = float(np.median(toggle_jumps))
    p90_db = float(np.percentile(toggle_jumps, 90.0))
    assert median_db < 1.5, f"ISB/RSB median total-field jump {median_db:.3f} dB >= 1.5"
    assert p90_db < 5.0, f"ISB/RSB p90 total-field jump {p90_db:.3f} dB >= 5.0"


def test_t2_extension_plane_continuity() -> None:
    """Assert total-field continuity where diffraction crosses an extended-face plane."""
    _require_cuda()
    x = np.round(np.arange(-0.010, 0.0041, 0.0005), 6)
    y = np.round(np.array([0.6840, 0.6840 + 0.0005]), 6)
    field, _ = _solve_maps(x, y, {"los", "diffraction"}, max_depth=1)

    assert np.isfinite(field).all(), "extension-plane line contains NaN/Inf"
    field_db = _db(np.abs(field))
    max_jump_x = float(np.abs(field_db[:, 1:] - field_db[:, :-1]).max())
    max_jump_y = float(np.abs(field_db[1:, :] - field_db[:-1, :]).max())
    max_jump = max(max_jump_x, max_jump_y)
    assert max_jump < 2.5, f"extension-plane max adjacent jump {max_jump:.3f} dB >= 2.5"


def test_t3_near_edge_sliver_is_alive() -> None:
    """Assert near-edge diffraction stays finite without a hard distance cutoff."""
    _require_cuda()
    x = np.round(np.arange(0.098, 0.1321, 0.0015), 6)
    y = np.round(np.arange(-0.102, -0.0459, 0.0015), 6)
    field, component = _solve_maps(x, y, _FULL_COMPONENTS, max_depth=2)
    valid = _footprint_valid(x, y)

    assert np.isfinite(field).all(), "near-edge sliver total field has NaN/Inf"
    diffraction_magnitude = np.abs(component["diffraction"])
    assert bool(valid.any()), "patch has no valid (non-footprint) cells"
    sliver_min = float(diffraction_magnitude[valid].min())
    assert sliver_min > 1.0e-5, (
        f"near-edge sliver min |diffraction| {sliver_min:.3e} <= 1e-5 "
        "(R1 5 cm gate regression)"
    )


def test_t4_vertex_ray_no_double_count() -> None:
    """Assert a vertex shadow-boundary ray is counted once."""
    _require_cuda()
    x = np.round(np.arange(-0.03, 0.0301, 0.0015), 6)
    y = np.round(np.arange(0.66, 0.7201, 0.0015), 6)
    field, component = _solve_maps(x, y, {"los", "diffraction"}, max_depth=1)

    assert np.isfinite(field).all(), "vertex patch total field has NaN/Inf"
    los = np.abs(component["los"])
    diffraction = np.abs(component["diffraction"])
    ratio = diffraction / np.maximum(los, 1.0e-30)
    # Corner-cone ridge = the peak-diffraction cell in each row (verify-e1e check4).
    ridge = np.array(
        [ratio[row, int(np.argmax(diffraction[row]))] for row in range(len(y))]
    )
    ridge_median = float(np.median(ridge))
    assert 0.2 <= ridge_median <= 0.8, (
        f"vertex ridge |diffraction_z|/|los_z| median {ridge_median:.3f} "
        "outside [0.2, 0.8] (double count or fake null)"
    )
    assert float(ridge.max()) <= 0.8, (
        f"vertex ridge ratio max {float(ridge.max()):.3f} > 0.8 (double count)"
    )

    field_db = _db(np.abs(field))
    max_jump_x = float(np.abs(field_db[:, 1:] - field_db[:, :-1]).max())
    max_jump_y = float(np.abs(field_db[1:, :] - field_db[:-1, :]).max())
    max_jump = max(max_jump_x, max_jump_y)
    assert max_jump <= 8.0, (
        f"vertex patch max adjacent |total| jump {max_jump:.3f} dB > 8.0 "
        "(corner-zone residual allowance)"
    )


def test_t5_no_dead_cells() -> None:
    """Assert no valid cell collapses while an adjacent valid cell remains strongly lit."""
    _require_cuda()
    x = np.linspace(-0.4, 0.4, 64)
    y = np.linspace(-0.4, 0.4, 64)
    field, _ = _solve_maps(x, y, _FULL_COMPONENTS, max_depth=2)
    valid = _footprint_valid(x, y)
    magnitude = np.abs(field)

    assert np.isfinite(field).all(), "full grid total field has NaN/Inf"
    assert float(magnitude[valid].max()) > 1.0e-4, "solve produced no lit field"

    dead = valid & (magnitude < 1.0e-9)
    hot = magnitude > 1.0e-4
    ny, nx = magnitude.shape
    dead_zero_pairs: list[tuple[int, int]] = []
    for i, j in np.argwhere(dead):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = int(i) + di, int(j) + dj
            if 0 <= ii < ny and 0 <= jj < nx and valid[ii, jj] and hot[ii, jj]:
                dead_zero_pairs.append((int(i), int(j)))
                break
    assert dead_zero_pairs == [], (
        f"{len(dead_zero_pairs)} dead-zero cell(s) next to a live neighbour: "
        f"{dead_zero_pairs[:5]}"
    )