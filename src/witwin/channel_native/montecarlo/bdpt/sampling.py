from __future__ import annotations

import torch

from witwin.channel_native.montecarlo.bdpt.kernels.paths import bdpt_launch_state

from .config import Config


def make_launch_state(reference: torch.Tensor, *, tx_count: int, config: Config) -> dict[str, torch.Tensor]:
    return bdpt_launch_state(
        reference,
        tx_count=tx_count,
        samples=config.samples,
        sample_streams=config.sample_streams,
        seed=config.seed,
    )
