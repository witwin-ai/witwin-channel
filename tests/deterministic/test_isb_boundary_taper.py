"""ISB boundary taper (ADR-017) LoS-member solver tests.

Gate 1 (OFF bit-identical) is exercised without a build-time baseline by
comparing the flag-absent default solve to an explicit ``isb_boundary_taper=
False`` solve: both take the untouched hard-gate path and must be byte-identical.
Gate 2 (ON softens the LoS shadow boundary) checks that the tapered solve adds
intermediate LoS amplitudes across a blocked boundary that the hard gate never
produces, and that the D-side window only engages when the flag is on.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from witwin.channel_native import ReceiverGrid, Scene, Structure, Transmitter
from witwin.channel_native.core.materials import PerfectConductor
from witwin.channel_native.deterministic import Config, solve


def _metal_box(center: tuple[float, float, float], half: float, name: str) -> Structure:
    cx, cy, cz = center
    lo = (cx - half, cy - half, cz - half)
    hi = (cx + half, cy + half, cz + half)
    vertices = torch.tensor(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # -y
            [1, 2, 6], [1, 6, 5],  # +x
            [2, 3, 7], [2, 7, 6],  # +y
            [3, 0, 4], [3, 4, 7],  # -x
        ],
        dtype=torch.int32,
    )
    return Structure(
        vertices=vertices, faces=faces, material=PerfectConductor(), name=name
    )


def _two_cube_grid_scene() -> Scene:
    # Transmitter above two metal cubes; a receiver grid on the z = 0 plane whose
    # rows sweep across the shadow boundaries the cubes cast, so both fully-lit,
    # fully-shadowed, and penumbra-margin cells appear.
    boxes = [
        _metal_box((-0.12, 0.0, 0.3), 0.1, "cube-a"),
        _metal_box((0.16, 0.0, 0.3), 0.1, "cube-b"),
    ]
    grid = ReceiverGrid(
        origin=torch.tensor([-0.45, 0.0, 0.0]),
        x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]),
        shape=(64, 1),
        spacing=(0.014, 0.014),
    )
    return Scene(
        structures=boxes,
        transmitters=[Transmitter(position=torch.tensor([-0.02, 0.0, 0.7]))],
        receivers=[grid],
        frequency=5.0e9,
    )


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the deterministic solver")


def test_isb_boundary_taper_off_is_bit_identical():
    _require_cuda()
    scene = _two_cube_grid_scene()
    components = {"los", "reflection", "diffraction"}
    # Flag absent (the default) vs explicit False: both take the untouched hard
    # occlusion gate and the unchanged order-1 diffraction window, so the taper
    # must be a perfect no-op (ADR-017 gate 1). The OFF reproducibility contract
    # has the two tiers spelled out in ADR-013's "bitwise-off note": the
    # deterministic per-row path table and the LoS/reflection component
    # accumulators are byte-identical, but the folded *total* field carries the
    # pre-existing float32-ULP atomic-order noise of the diffraction accumulation
    # (up to ~1e-9 run-to-run for a single build; auditing that reduction's
    # atomic-order determinism is the plan-09 P5 chore). We therefore assert the
    # per-row and LoS/reflection contracts bitwise and bound the total-field
    # delta to the documented 5e-10 band, rather than over-fitting to a
    # nondeterministic atomic sum (the old torch.equal on the total field / total
    # path_gain flaked on exactly this noise despite a perfect OFF no-op).
    default_result = solve(scene, Config(components=components, export_paths=True))
    explicit_off = solve(
        scene, Config(components=components, isb_boundary_taper=False, export_paths=True)
    )

    # (a) The exported path-table row arrays are the deterministic per-row
    # contract the frozen-SHA landing checks hash; every column must match bit
    # for bit.
    assert default_result.paths is not None
    assert explicit_off.paths is not None
    for column in dataclasses.fields(default_result.paths):
        assert torch.equal(
            getattr(default_result.paths, column.name),
            getattr(explicit_off.paths, column.name),
        ), f"path-table column {column.name!r} diverged with the taper flag off"

    # (b) The LoS and reflection component accumulators are proven bitwise
    # reproducible: they do not fold through the noisy diffraction atomics.
    for component in ("los", "reflection"):
        assert torch.equal(
            default_result.component_fields[component],
            explicit_off.component_fields[component],
        )
        assert torch.equal(
            default_result.component_power[component],
            explicit_off.component_power[component],
        )

    # (c) The total field folds the diffraction contribution through an
    # atomic-add reduction whose ordering is not run-to-run stable (ADR-013
    # bitwise-off note; plan-09 P5 chore). Bound its delta to the documented
    # 5e-10 absolute band instead of demanding exact equality.
    field_delta = (default_result.field - explicit_off.field).abs().max()
    assert field_delta <= 5e-10, (
        f"total-field delta {float(field_delta):.3e} exceeds the documented "
        "5e-10 diffraction atomic-order reproducibility band"
    )


def test_isb_boundary_taper_on_softens_los_shadow_boundary():
    _require_cuda()
    scene = _two_cube_grid_scene()
    components = {"los", "reflection", "diffraction"}
    off = solve(scene, Config(components=components, isb_boundary_taper=False))
    on = solve(
        scene,
        Config(
            components=components,
            isb_boundary_taper=True,
            isb_boundary_taper_width=0.5,
        ),
    )

    los_off = off.component_power["los"].reshape(-1)
    los_on = on.component_power["los"].reshape(-1)

    # The taper must actually change the LoS component (the boundary rows are
    # spread), not silently no-op.
    assert not torch.equal(los_off, los_on)

    # The hard gate produces only {0, lit} LoS power per cell. The taper adds
    # penumbra cells whose power is strictly between the shadow floor and the
    # neighbouring lit level: at least one cell that is exactly zero off becomes
    # a small-but-nonzero survivor on, or a lit cell is attenuated below its off
    # value. Either way the boundary is no longer a single-cell cliff.
    lit_level = float(los_off.max())
    assert lit_level > 0.0
    newly_survived = ((los_off == 0.0) & (los_on > 0.0)).any()
    attenuated_lit = ((los_off > 0.0) & (los_on < los_off - 1e-12)).any()
    assert bool(newly_survived) or bool(attenuated_lit)

    # The largest adjacent-cell LoS amplitude step (dB) across the boundary must
    # be smaller with the taper on than the hard-gate cliff.
    def _max_adjacent_db_jump(power: torch.Tensor) -> float:
        amp_db = 10.0 * torch.log10(power.clamp_min(1e-30))
        return float((amp_db[1:] - amp_db[:-1]).abs().max())

    # The off boundary steps from a finite lit level straight to the shadow
    # floor (a very large dB jump); the taper must reduce the worst step.
    assert _max_adjacent_db_jump(los_on) < _max_adjacent_db_jump(los_off)
