"""Deterministic-field continuity regression guards (design doc Tier 3 / F6).

Companion to ``docs/dev/audit/utd-continuity-fix-design.md`` (section F6) and
``docs/dev/audit/fullwave-deterministic-discontinuity-audit.md``. These are
deterministic-only, GPU, no-fullwave-reference tests over the single-cube PEC
scene (cube center (0,0,0.15), size 0.2 m, PEC; TX (-0.2,-0.5,0.42) z-pol;
receiver plane z=0.10 m; 5 GHz). They pin the R1-R5 continuity fixes so a
future regression that re-introduces a hard gate (R1), a dominant-component
export collapse (R2), a receiver-side coefficient branch (R3), a vertex
double-count (R4) or a polarization-model mismatch (R5) fails loudly.

Every threshold is calibrated from the verified E1e build metrics
(``artifacts/fullwave-fix/verify-e1e/verify_e1e_metrics.json`` and the
``check{1..4}_*`` scripts, pyd fingerprint
ec59cfbf202c921b8cff4dc827a38db201fa3926a62cfc1f53d756c19dade90a) and set with
margin over the value measured on each test's own (small) grid; the per-test
provenance is documented inline. The deterministic solver is reproducible
(bit-identical across repeated solves), so these thresholds are not flaky.

Cells inside the cube footprint (|x| <= 0.1 and |y| <= 0.1) lie inside the PEC
on the z=0.10 plane and legitimately carry no field; they are excluded from the
continuity statistics via ``_footprint_valid`` exactly as the verify-e1e full-
grid checks do (``check3_fullgrid.footprint_valid``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from witwin.channel_native import (
    PerfectConductor,
    ReceiverGrid,
    Scene,
    Structure,
    Transmitter,
)
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.deterministic import Config, solve


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
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
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
    structure = Structure(
        vertices=vertices,
        faces=faces,
        material=PerfectConductor(name="pec"),
        name="cube-1",
        surface_id=1,
    )
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    grid = ReceiverGrid(
        origin=torch.tensor([float(x[0]), float(y[0]), _PLANE_Z]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(x.size, y.size),
        spacing=(dx, dy),
        polarization=torch.tensor(list(_RX_POLARIZATION)),
    )
    return Scene(
        structures=[structure],
        transmitters=[
            Transmitter(
                position=torch.tensor(list(_TX_POSITION)),
                polarization=torch.tensor(list(_TX_POLARIZATION)),
            )
        ],
        receivers=[grid],
        frequency=_FREQUENCY_HZ,
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
    """F1/F3/F4/F5 (R2/R3/R4): the total field is continuous across GO toggles.

    At every LoS-support toggle (ISB) and reflection-support toggle (RSB) the
    diffracted field must compensate the geometrical-optics step, so the total
    |field| dB jump across the toggle pair stays small.

    Provenance: verify-e1e G1 measured on the 256x256 grid ISB median 1.146 dB /
    p90 2.85 dB, RSB median 0.687 dB / p90 3.10 dB
    (verify_e1e_metrics.json G1_los_refl_toggle_fullgrid, check2). On this
    coarser 64x64 grid the combined ISB+RSB toggle set measures median 1.097 dB
    and p90 3.85 dB (reproducible). Thresholds 1.5 dB / 5.0 dB keep ~27% / ~23%
    margin. A regression to the R1 hard gate or the R2 dominant-component
    collapse re-inflates these by many dB.
    """
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
    """F5c (R3a): the total field is continuous across an extended-face plane.

    Line y=0.6840, x in [-0.010, 0.004] at 0.5 mm (2 rows). The NW-vertical edge
    diffraction crosses its extension plane at x~-0.00267; the closed-form
    corner mend keeps the LoS+diffraction total continuous there.

    Provenance: verify-e1e G2 (check1) measured 0.651 dB on this exact line
    (gate was <2.0 dB; E1d without the corner mend was 3.585 dB). This 2-row
    grid measures 0.654 dB. Threshold 2.5 dB keeps ~3.8x margin over the
    measured step while still failing the 3.585 dB pre-mend regression.
    """
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
    """F2 (R1): removing the 5 cm gate keeps near-edge diffraction finite.

    Patch x in [0.098, 0.132], y in [-0.102, -0.046] at 1.5 mm, just east of the
    +x cube face where the old UTD_MIN_DISTANCE=5e-2 gate hard-zeroed the
    stationary-point contribution. The diffraction component must stay finite
    over the valid (footprint-excluded) cells.

    Provenance: verify-e1e G3 (check3) dead-sliver min |diffraction| 1.159e-3
    (pre-fix baseline 2.2e-10). On this patch the footprint-excluded minimum is
    2.34e-4. Threshold 1e-5 keeps >20x margin over the measured floor and >4
    orders of magnitude over the pre-fix hard zero. Footprint-interior cells are
    inside the PEC and legitimately zero, so they are excluded.
    """
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
    """F5 (R4): a vertex-generated shadow-boundary ray is not double-counted.

    Patch x in [-0.03, 0.03], y in [0.66, 0.72] at 1.5 mm, over the corner-cone
    ridge cast by the NW-top cube vertex. With the truncated-edge sum each edge
    contributes ~E_i/4 and the ridge ratio |diffraction_z|/|los_z| sits near
    E_i/4..E_i/2 rather than ~1 (the pre-fix fake null where diffraction ~= los
    in antiphase, or a >1 double count).

    Provenance: verify-e1e check4 vertex ridge ratio median 0.344
    (require [0.25, 0.75]) and worst adjacent |total_z| jump 4.903 dB at 0.5 mm
    (documented corner-zone residual, F5c future work). On this 1.5 mm grid the
    ridge ratio is 0.343 (min 0.305, max 0.346) and the worst adjacent total-z
    jump is 1.76 dB. Ratio band [0.2, 0.8] brackets the corner limit without a
    double count; the 8 dB adjacent-jump allowance documents the residual
    corner-cone step (still far below a fake-null collapse).
    """
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
    """R1/R2: no spuriously dead cell next to a live one on the full grid.

    Full 64x64 grid: no valid (footprint-excluded) receiver may collapse below
    1e-9 while an orthogonal valid neighbour exceeds 1e-4. Such a pair is the
    signature of a hard gate (R1) or an export collapse (R2) zeroing an
    otherwise-lit cell; a genuine coherent null is never that deep on this grid.

    Provenance: verify-e1e full grid has zero NaN/Inf and the deepest physical
    null cell sits at |E|~1.5e-4 (verify_e1e full_grid_jumps_seams), while the
    pre-fix dead sliver was ~2.2e-10. On this grid the minimum valid magnitude
    is 2.25e-5 (>> 1e-9), so the detector finds no dead-zero/live-neighbour pair.
    """
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
