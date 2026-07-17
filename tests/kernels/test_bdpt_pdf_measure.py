from __future__ import annotations

import pytest
import torch

from witwin.channel_native.montecarlo.bdpt.kernels import paths as ops
from witwin.channel_native.montecarlo.events.scattering import (
    solid_angle_to_area_jacobian,
)


def test_solid_angle_to_area_jacobian_is_explicit_geometry():
    cosine = torch.tensor([1.0, 0.5, -0.25])
    distance = torch.tensor([2.0, 4.0, 8.0])
    jacobian = solid_angle_to_area_jacobian(cosine, distance)

    torch.testing.assert_close(jacobian, torch.tensor([0.25, 0.03125, 0.0]))
    assert torch.isfinite(jacobian).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for native BDPT PDF audit"
)
def test_endpoint_proposal_pdf_excludes_inverse_square_geometry():
    tx = torch.tensor([[0.0, 0.0, 0.0]], device="cuda")
    power = torch.ones((1,), device="cuda")
    polarization = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    tx_id = torch.zeros((1,), device="cuda", dtype=torch.int32)
    seed = torch.tensor([7], device="cuda", dtype=torch.int64)

    def connect(distance: float) -> dict[str, torch.Tensor]:
        endpoints = ops.bdpt_endpoint_subpath_state(
            tx,
            power,
            polarization,
            torch.tensor([[distance, 0.0, 0.0]], device="cuda"),
            polarization,
            tx_id,
            seed,
        )
        return ops.bdpt_endpoint_connection_samples(
            endpoints["light"],
            endpoints["sensor"],
            frequency_hz=3.0e9,
            samples_per_tx=1,
            strategy_count=1,
        )

    near = connect(1.0)
    far = connect(2.0)
    torch.testing.assert_close(near["pdf"], far["pdf"], rtol=0.0, atol=0.0)
    assert float(near["contribution"] / far["contribution"]) == pytest.approx(4.0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for native BDPT MIS audit"
)
def test_endpoint_rejects_fake_multistrategy_count():
    reference = torch.empty((1, 3), device="cuda")
    state = ops.bdpt_empty_subpath_state(reference)
    with pytest.raises(ValueError, match="exactly one strategy"):
        ops.bdpt_endpoint_connection_samples(
            state,
            state,
            frequency_hz=3.0e9,
            samples_per_tx=1,
            strategy_count=2,
        )


def test_mis_weights_are_bounded_for_positive_proposals():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native BDPT MIS audit")
    pdf = torch.tensor([0.25, 0.75], device="cuda")
    weights = ops.bdpt_mis_weights(
        pdf,
        torch.tensor(1.0, device="cuda"),
        mis="balance",
    )
    assert bool(((weights >= 0.0) & (weights <= 1.0)).all())
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, device="cuda"))
