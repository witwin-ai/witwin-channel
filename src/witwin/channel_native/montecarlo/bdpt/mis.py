from __future__ import annotations

import torch

from witwin.channel_native.montecarlo.bdpt.kernels.paths import bdpt_mis_weights


def compute_mis_weights(
    pdf: torch.Tensor,
    *,
    strategy_pdf_sum: torch.Tensor,
    mis: str,
    beta: float = 2.0,
) -> torch.Tensor:
    return bdpt_mis_weights(pdf, strategy_pdf_sum, mis=mis, beta=beta)


def mis_mode_id(mis: str) -> int:
    if mis == "none":
        return 0
    if mis == "balance":
        return 1
    if mis == "power_heuristic":
        return 2
    raise ValueError("mis is not supported")
