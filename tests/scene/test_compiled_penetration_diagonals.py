# Copyright Xingyu Chen.
# Tests compiled penetration diagonals.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from witwin.core import PhysicalMaterial

from witwin.channel.scene.compiler import (
    _compile_materials,
    _compile_penetration_scene_diagonals,
    _unique_materials,
)


class _RayDRecords:
    def __init__(self, vertices: torch.Tensor) -> None:
        self._records = SimpleNamespace(vertices=vertices)

    def edge_records(self) -> object:
        return self._records


def test_compile_freezes_distinct_enumerated_and_montecarlo_diagonals() -> None:
    structures = (
        SimpleNamespace(
            vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32
            )
        ),
        SimpleNamespace(
            vertices=torch.tensor(
                [[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float32
            )
        ),
    )
    rayd = _RayDRecords(
        torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    )

    enumerated, montecarlo = _compile_penetration_scene_diagonals(  # type: ignore[arg-type]
        structures,
        rayd=rayd,  # type: ignore[arg-type]
    )

    assert enumerated == pytest.approx(12.0**0.5)
    assert montecarlo == pytest.approx(29.0**0.5)


def test_empty_scene_has_zero_penetration_diagonals_without_rayd_read() -> None:
    class _NoRead:
        def edge_records(self) -> object:
            raise AssertionError("empty scene must not query RayD records")

    assert _compile_penetration_scene_diagonals((), rayd=_NoRead()) == (0.0, 0.0)  # type: ignore[arg-type]


def test_material_store_uses_core_structure_material_ids() -> None:
    logical = _unique_materials(  # type: ignore[arg-type]
        (
            SimpleNamespace(
                material_id=977,
                material=PhysicalMaterial(eps_r=4.0),
            ),
        )
    )

    store = _compile_materials(logical, 2.4e9, 0)

    assert store.material_id.tolist() == [977]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_accepts_mixed_cuda_and_cpu_structure_vertices() -> None:
    structures = (
        SimpleNamespace(
            vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                dtype=torch.float32,
                device="cuda",
                requires_grad=True,
            )
        ),
        SimpleNamespace(
            vertices=torch.tensor(
                [[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=torch.float32
            )
        ),
    )
    rayd = _RayDRecords(
        torch.tensor(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
            device="cuda",
        )
    )

    enumerated, montecarlo = _compile_penetration_scene_diagonals(  # type: ignore[arg-type]
        structures,
        rayd=rayd,  # type: ignore[arg-type]
    )

    assert enumerated == pytest.approx(12.0**0.5)
    assert montecarlo == pytest.approx(29.0**0.5)