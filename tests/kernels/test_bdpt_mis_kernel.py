import pytest
import torch

from witwin.channel_native.core.kernels import ops
from witwin.channel_native.montecarlo.bdpt.kernels import paths


def test_bdpt_mis_kernel_matches_expected_constants():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT MIS kernel")

    pdf = torch.tensor([0.0, 0.25, 1.0, 2.0], device="cuda", dtype=torch.float32)
    strategy_sum = torch.tensor(5.0625, device="cuda", dtype=torch.float32)

    weights = ops.bdpt_mis_weights(pdf, strategy_sum, mis="power_heuristic", beta=2.0)

    assert weights.detach().cpu().tolist() == pytest.approx([0.0, 0.012345679, 0.197530864, 0.790123456])


def test_bdpt_mis_kernel_validates_cuda_inputs():
    pdf = torch.ones(1, dtype=torch.float32)
    strategy_sum = torch.ones((), dtype=torch.float32)

    with pytest.raises(ValueError, match="pdf must be a CUDA tensor"):
        ops.bdpt_mis_weights(pdf, strategy_sum, mis="balance")


def test_bdpt_mis_kernel_has_no_python_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BDPT MIS kernel")

    pdf = torch.ones(1, device="cuda", dtype=torch.float32)
    strategy_sum = torch.ones((), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(paths, "native_extension", lambda: None)

    with pytest.raises(RuntimeError, match="bdpt_mis_weights CUDA kernel is required"):
        ops.bdpt_mis_weights(pdf, strategy_sum, mis="balance")
