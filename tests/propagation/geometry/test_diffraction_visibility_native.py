from __future__ import annotations

import pytest
import torch

from tests.support.scenes import wedge_diffraction_scene
from witwin.channel.core.kernels.extension import build_info
from witwin.channel.propagation.enumerated.diffraction import (
    _deterministic_diffraction_states,
)
from witwin.channel.propagation.geometry import diffraction
from witwin.channel.propagation.geometry.kernels import bridge


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_tx_visibility_plan_matches_frozen_four_sample_contract() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction is not built")

    scene = wedge_diffraction_scene()
    compiled = scene.compile()
    rayd = compiled.rayd
    tx = scene.transmitters[0].position.to(device="cuda", dtype=torch.float32)
    tx = tx.contiguous()
    states = _deterministic_diffraction_states(
        rayd,
        tx,
        torch.ones(1, device="cuda"),
        0,
        preserve_imported_edges=True,
    )
    assert int(states[0].shape[0]) > 0

    expected = torch.zeros(states[0].shape, device="cuda", dtype=torch.bool)
    starts = tx.reshape(1, 3).expand(int(states[0].shape[0]), 3).contiguous()
    span = states[4] - states[3]
    for fraction in (0.02, 1.0 / 3.0, 2.0 / 3.0, 0.98):
        scaled = span * fraction
        sample_t = states[3] + scaled
        offset = sample_t.unsqueeze(1) * states[2]
        points = (states[1] + offset).contiguous()
        visible = bridge.rayd_visibility_forward(
            rayd.require_resource(), starts, points, None
        )[0]
        expected |= visible

    plan = diffraction.plan_tx_visible_diffraction_states(rayd, states, tx)

    assert plan.active.is_cuda
    assert plan.active.dtype == torch.bool
    assert plan.active.is_contiguous()
    assert plan.active.shape == states[0].shape
    assert torch.equal(plan.active, expected)
    assert all(
        value is states[index]
        for index, value in enumerate(
            (
                plan.edge_index,
                plan.edge_position,
                plan.edge_direction,
                plan.edge_t_min,
                plan.edge_t_max,
                plan.n0,
                plan.n1,
                plan.prim0,
                plan.prim1,
                plan.exterior_angle,
                plan.source,
                plan.source_power,
            )
        )
    )

    mismatched_source = torch.full_like(states[10], 123.0)
    source_mismatch_states = (*states[:10], mismatched_source, states[11])
    source_mismatch_plan = diffraction.plan_tx_visible_diffraction_states(
        rayd, source_mismatch_states, tx
    )
    assert source_mismatch_plan.source is mismatched_source
    assert torch.equal(source_mismatch_plan.active, plan.active)

    empty_states = tuple(tensor[:0] for tensor in states)
    empty_plan = diffraction.plan_tx_visible_diffraction_states(
        rayd, empty_states, tx
    )
    assert empty_plan.active.is_cuda
    assert empty_plan.active.dtype == torch.bool
    assert empty_plan.active.shape == (0,)
    assert empty_plan.active.is_contiguous()
