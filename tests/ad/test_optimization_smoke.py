"""End-to-end optimization smoke (plan 07 section 9.1, last bullet).

Adam recovers a transmitter position through the deterministic solver's
differentiable outputs from a perturbed start. AD-3 already demonstrated
material recovery through the montecarlo.basic power map; this is the
endpoint counterpart for the D/P track.

The loss is a time-of-arrival (delay) match over four receivers -- a
multilateration objective that is smooth and phase-wrap free, so it
exercises the geometry chain (live endpoints -> differentiable path
delays) without the lambda/2 local minima a complex-coefficient loss would
add at 3 GHz. Path delays became differentiable outputs in AD-2 exactly for
this class of loss.
"""

from __future__ import annotations

import pytest
import torch

from witwin.channel import ReceiverPoint, Scene, Transmitter
from witwin.channel.deterministic import Config, solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the optimization smoke"
)

_FREQUENCY_HZ = 3.0e9
_TX_TRUE = (0.4, -0.7, 0.6)
_TX_START_OFFSET = (0.35, -0.3, 0.25)
_RX_POSITIONS = (
    (3.0, 1.0, 0.2),
    (-2.0, 2.0, 1.0),
    (1.5, -3.0, 1.4),
    (-1.0, -2.0, -0.5),
)
_STEPS = 200
_POSITION_TOLERANCE_M = 1.0e-2


def _scene(tx: torch.Tensor) -> Scene:
    return Scene(
        structures=[],
        transmitters=[Transmitter(position=tx)],
        receivers=[
            ReceiverPoint(position=torch.tensor(position))
            for position in _RX_POSITIONS
        ],
        frequency=_FREQUENCY_HZ,
    )


def _delays(tx: torch.Tensor, ad_mode: str) -> torch.Tensor:
    result = solve(
        _scene(tx),
        Config(
            max_depth=0,
            components={"los"},
            export_paths=True,
            ad_mode=ad_mode,
        ),
    )
    delays = result.paths.delay_s
    assert int(delays.shape[0]) == len(_RX_POSITIONS)
    return delays


def test_adam_recovers_transmitter_position():
    tx_true = torch.tensor(_TX_TRUE, device="cuda")
    with torch.no_grad():
        target = _delays(tx_true, "none").detach()

    tx = (
        tx_true + torch.tensor(_TX_START_OFFSET, device="cuda")
    ).clone().requires_grad_(True)
    start_error = float((tx.detach() - tx_true).norm())
    optimizer = torch.optim.Adam([tx], lr=2.0e-2)
    # Delays are ~1e-8 s; scale the residual so Adam sees O(1) losses.
    scale = 1.0e9
    for _ in range(_STEPS):
        optimizer.zero_grad()
        loss = ((_delays(tx, "vjp") - target) * scale).square().sum()
        loss.backward()
        assert tx.grad is not None and bool(torch.isfinite(tx.grad).all())
        optimizer.step()

    final_error = float((tx.detach() - tx_true).norm())
    assert final_error < _POSITION_TOLERANCE_M, (
        f"Adam did not recover the transmitter: start {start_error:.3f} m, "
        f"final {final_error:.4f} m"
    )
    assert final_error < 0.1 * start_error
