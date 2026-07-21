from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel_native.scene.compile import _compile_penetration_scene_diagonals


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
        torch.tensor(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32
        )
    )

    enumerated, montecarlo = _compile_penetration_scene_diagonals(  # type: ignore[arg-type]
        structures, rayd=rayd  # type: ignore[arg-type]
    )

    assert enumerated == pytest.approx(12.0**0.5)
    assert montecarlo == pytest.approx(29.0**0.5)


def test_empty_scene_has_zero_penetration_diagonals_without_rayd_read() -> None:
    class _NoRead:
        def edge_records(self) -> object:
            raise AssertionError("empty scene must not query RayD records")

    assert _compile_penetration_scene_diagonals((), rayd=_NoRead()) == (0.0, 0.0)  # type: ignore[arg-type]
