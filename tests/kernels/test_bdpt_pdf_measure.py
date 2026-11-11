# Copyright Xingyu Chen.
# Tests bdpt pdf measure.

from __future__ import annotations

import pytest
import torch

from witwin.channel.kernels import montecarlo as ops


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

