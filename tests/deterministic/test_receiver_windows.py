# Copyright Xingyu Chen.
# A coupled solve over many receivers in receiver windows equals the whole solve.

from __future__ import annotations

import pytest
import torch

import witwin.channel.deterministic as deterministic
from tests.support.core_world import make_receiver_grid, make_transmitter
from tests.support.scenes import coupled_wall_wedge_scene
from witwin.channel.deployment import build_info
from witwin.core import Scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")


def _scene() -> Scene:
    base = coupled_wall_wedge_scene()
    grid = make_receiver_grid(
        origin=torch.tensor([-3.0, -3.0, 1.5]), x_axis=torch.tensor([1.0, 0.0, 0.0]),
        y_axis=torch.tensor([0.0, 1.0, 0.0]), shape=(48, 40), spacing=(0.125, 0.15),
    )
    return Scene(structures=base.structures, endpoints=[make_transmitter(torch.tensor([0.0, -2.0, 1.0])), grid])


def _solve(scene: Scene, **overrides) -> deterministic.Result:
    config = deterministic.Config(
        components=frozenset({"los", "reflection", "diffraction", "transmission"}), max_depth=3,
        coupled_paths=True, **overrides)
    return deterministic.solve(scene, config, reference_frequency_hz=3.0e9)


def test_windowed_coupled_solve_equals_the_whole_solve(monkeypatch):
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")
    scene = _scene()
    whole = _solve(scene)
    assert int(whole.path_gain.numel()) == 48 * 40
    # Small windows: the first holds 300 receivers, later ones about 2000 rows each.
    monkeypatch.setattr(deterministic, "_RECEIVER_WINDOW", 300)
    monkeypatch.setattr(deterministic, "_WINDOW_ROWS", 2000)
    windowed = _solve(scene)
    assert windowed.metadata["counts"]["path_count"] == whole.metadata["counts"]["path_count"]
    assert windowed.metadata["counts"] == whole.metadata["counts"]
    assert windowed.metadata["kernel"]["launch_count"] > whole.metadata["kernel"]["launch_count"]
    # Each receiver's rows are the whole solve's, summed in one window: float atomics only.
    torch.testing.assert_close(windowed.path_gain, whole.path_gain, rtol=1e-5, atol=0.0)
    torch.testing.assert_close(windowed.field, whole.field, rtol=1e-5, atol=1e-12)
    assert set(windowed.component_fields) == set(whole.component_fields)
    for name, value in whole.component_fields.items():
        torch.testing.assert_close(windowed.component_fields[name], value, rtol=1e-5, atol=1e-12)
        torch.testing.assert_close(windowed.component_power[name], whole.component_power[name],
                                   rtol=1e-5, atol=1e-20)
    assert float(whole.component_power["coupled"].abs().sum()) > 0.0


def test_solves_that_need_every_row_are_not_windowed(monkeypatch):
    monkeypatch.setattr(deterministic, "_RECEIVER_WINDOW", 4)
    config = deterministic.Config(components=frozenset({"los", "reflection", "diffraction"}),
                                  max_depth=2, coupled_paths=True)
    assert deterministic._windowed(config, 100)
    assert not deterministic._windowed(config, 4)
    for change in ({"export_paths": True}, {"max_paths": 10}, {"ad_mode": "jvp"}, {"coupled_paths": False}):
        fields = {"components": frozenset({"los", "reflection", "diffraction"}), "max_depth": 2,
                  "coupled_paths": True, **change}
        assert not deterministic._windowed(deterministic.Config(**fields), 100), change
