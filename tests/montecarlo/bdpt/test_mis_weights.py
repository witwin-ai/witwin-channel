import pytest
import torch

from witwin.channel.montecarlo.bdpt.mis import compute_mis_weights


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native BDPT MIS")


def test_mis_none_returns_unit_weight_for_positive_pdf():
    pdf = torch.tensor([0.0, 0.25, 2.0], device="cuda", dtype=torch.float32)

    weights = compute_mis_weights(
        pdf,
        strategy_pdf_sum=torch.tensor(2.25, device="cuda", dtype=torch.float32),
        mis="none",
    )

    assert weights.detach().cpu().tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_balance_heuristic_normalizes_pdf_against_strategy_sum():
    pdf = torch.tensor([1.0, 3.0], device="cuda", dtype=torch.float32)

    weights = compute_mis_weights(
        pdf,
        strategy_pdf_sum=torch.tensor(4.0, device="cuda", dtype=torch.float32),
        mis="balance",
    )

    assert weights.detach().cpu().tolist() == pytest.approx([0.25, 0.75])


def test_power_heuristic_uses_configurable_beta():
    pdf = torch.tensor([1.0, 3.0], device="cuda", dtype=torch.float32)

    weights = compute_mis_weights(
        pdf,
        strategy_pdf_sum=torch.tensor(10.0, device="cuda", dtype=torch.float32),
        mis="power_heuristic",
        beta=2.0,
    )

    assert weights.detach().cpu().tolist() == pytest.approx([0.1, 0.9])


def test_mis_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mis"):
        compute_mis_weights(
            torch.ones(1, device="cuda", dtype=torch.float32),
            strategy_pdf_sum=torch.ones((), device="cuda", dtype=torch.float32),
            mis="bad",
        )
