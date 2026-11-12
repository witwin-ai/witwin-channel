# Copyright Xingyu Chen.
# Scene metadata does not select the diffraction edge table; every scene uses the merged wedges.

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
import torch
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

from tests.accuracy.scenes import antenna, grid

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")

FREQ = 3.5e9
# A Mitsuba metadata key the solvers ignore: it must not select RayD's raw edge records.
LEGACY_METADATA = {"mitsuba": {"merge_shapes": True}}


def _structure(vertices: list, faces: list, material: PhysicalMaterial) -> Structure:
    mesh = Mesh(torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32),
                recenter=False, fill_mode="surface", topology_diagnostics=False)
    return Structure(mesh, material)


def _box(lo: tuple, hi: tuple) -> tuple[list, list]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    vertices = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return vertices, faces


def _scene(metadata: dict[str, Any] | None) -> Scene:
    """A block on a ground block and two plates joined along an edge: both make coincident edges."""

    concrete = PhysicalMaterial(eps_r=5.3, sigma_e=0.03)
    pec = PhysicalMaterial.perfect_conductor()
    plate_a = [[4.0, -2.0, 0.0], [6.0, -2.0, 0.0], [6.0, 2.0, 0.0], [4.0, 2.0, 0.0]]
    plate_b = [[6.0, -2.0, 0.0], [6.0, 2.0, 0.0], [6.0, 2.0, 2.0], [6.0, -2.0, 2.0]]
    plate_faces = [[0, 1, 2], [0, 2, 3]]
    structures = [
        _structure(*_box((-1.0, -1.0, 0.0), (1.0, 1.0, 3.0)), concrete),
        _structure(*_box((-12.0, -12.0, -3.0), (12.0, 12.0, 0.0)), concrete),
        _structure(plate_a, plate_faces, pec),
        _structure(plate_b, plate_faces, pec),
    ]
    endpoints = [antenna("tx", (-6.0, -4.0, 4.0)), grid((-5.0, 2.0, 1.5), (6, 6), 2.0)]
    kwargs = {} if metadata is None else {"metadata": metadata}
    return Scene(structures=structures, endpoints=endpoints, **kwargs)


def _assert_same_tensors(left: Any, right: Any, label: str) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), label
        assert left.dtype == right.dtype and left.shape == right.shape, label
        # Bitwise: compare the raw bits, so NaN payloads in unused slots compare equal to themselves.
        if left.is_floating_point() or left.is_complex():
            bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[left.element_size() // (
                2 if left.is_complex() else 1)]
            left = torch.view_as_real(left) if left.is_complex() else left
            right = torch.view_as_real(right) if right.is_complex() else right
            left, right = left.contiguous().view(bits), right.contiguous().view(bits)
        assert torch.equal(left, right), f"{label} differs"
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), label
        for key in left:
            _assert_same_tensors(left[key], right[key], f"{label}[{key!r}]")


def test_legacy_mitsuba_metadata_does_not_select_another_edge_table():
    from witwin.channel import scene as channel_scene
    from witwin.channel.interactions.diffraction import selected_diffraction_edges
    from witwin.channel.propagation.geometry import diffraction_edge_geometry

    plain = channel_scene.compile(_scene(None), reference_frequency_hz=FREQ)
    legacy = channel_scene.compile(_scene(LEGACY_METADATA), reference_frequency_hz=FREQ)
    plain_table, plain_selected = selected_diffraction_edges(plain.rayd)
    legacy_table, legacy_selected = selected_diffraction_edges(legacy.rayd)

    assert torch.equal(plain_selected, legacy_selected)
    for field in fields(plain_table):
        _assert_same_tensors(
            getattr(plain_table, field.name), getattr(legacy_table, field.name), field.name
        )
    # The scene exercises the merge: RayD's raw records select the block's base edges and both
    # plates' junction edges, which the merged wedge table drops or folds into one wedge.
    raw_selected = diffraction_edge_geometry(legacy.rayd.edge_records())[0]
    assert int(legacy_selected.numel()) < int(raw_selected.sum())


@pytest.mark.parametrize("combine", ["field", "power"])
def test_legacy_mitsuba_metadata_leaves_coherent_paths_and_maps_bit_identical(combine):
    from witwin.channel.radiomap import Coherent, solve

    config = Coherent(
        max_depth=2,
        components={"los", "reflection", "diffraction"},
        coupled_paths=True,
        combine=combine,
        export_paths=True,
    )
    plain = solve(_scene(None), config, reference_frequency_hz=FREQ)
    legacy = solve(_scene(LEGACY_METADATA), config, reference_frequency_hz=FREQ)

    assert plain.paths is not None and legacy.paths is not None
    assert bool((plain.paths.component_id == 2).any())  # diffraction rows are present
    for field in fields(plain.paths):
        _assert_same_tensors(
            getattr(plain.paths, field.name), getattr(legacy.paths, field.name), f"paths.{field.name}"
        )
    _assert_same_tensors(plain.path_gain, legacy.path_gain, "path_gain")
    _assert_same_tensors(plain.field, legacy.field, "field")
    _assert_same_tensors(plain.component_power, legacy.component_power, "component_power")
    _assert_same_tensors(plain.component_fields, legacy.component_fields, "component_fields")


def test_legacy_mitsuba_metadata_leaves_the_stochastic_map_bit_identical():
    from witwin.channel.radiomap import Stochastic, solve

    config = Stochastic(samples=4096, max_depth=1, seed=7, components={"los", "diffraction"})
    plain = solve(_scene(None), config, reference_frequency_hz=FREQ)
    legacy = solve(_scene(LEGACY_METADATA), config, reference_frequency_hz=FREQ)

    assert float(plain.component_power["diffraction"].sum()) > 0.0
    _assert_same_tensors(plain.path_gain, legacy.path_gain, "path_gain")
    _assert_same_tensors(plain.component_power, legacy.component_power, "component_power")
