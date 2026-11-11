# Copyright Xingyu Chen.
# Tests that the selected-edge diagnostic is counted only when diagnostics ask for it.

from __future__ import annotations

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config, solve
from witwin.channel.kernels import topology as topology_kernels


def test_selected_edge_count_runs_only_for_diagnostics(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the deterministic solver")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")
    native = topology_kernels.deterministic_selected_edge_count
    calls = []

    def counted(edge_id):
        calls.append(int(edge_id.numel()))
        return native(edge_id)

    monkeypatch.setattr(topology_kernels, "deterministic_selected_edge_count", counted)
    config = dict(components={"los", "diffraction"}, max_depth=1)
    solve(wedge_diffraction_scene(), Config(**config), reference_frequency_hz=3.0e9)
    assert calls == []
    result = solve(
        wedge_diffraction_scene(), Config(**config, diagnostics=True), reference_frequency_hz=3.0e9
    )
    assert len(calls) == 1
    assert result.diagnostics["selected_edge_count"] > 0
