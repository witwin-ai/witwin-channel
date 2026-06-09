import pytest
import torch

from witwin.channel_native.core.kernels import ops


def test_validate_cuda_tensor_accepts_matching_cuda_tensor_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for positive CUDA tensor validation")

    tensor = torch.zeros((2, 3), device="cuda", dtype=torch.float32)

    assert ops.validate_cuda_tensor(
        "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    ) is tensor


def test_validate_cuda_tensor_rejects_cpu_tensor():
    tensor = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="points must be a CUDA tensor"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_validate_cuda_tensor_rejects_wrong_dtype_before_shape():
    tensor = torch.zeros((2, 3), dtype=torch.float64)

    with pytest.raises(TypeError, match="points must have dtype torch.float32"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_validate_cuda_tensor_rejects_trailing_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to isolate shape validation after device validation")

    tensor = torch.zeros((2, 2), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="points must end with shape"):
        ops.validate_cuda_tensor(
            "points", tensor, dtype=torch.float32, ndim=2, trailing_shape=(3,)
        )


def test_noop_metadata_reports_valid_schema():
    metadata = ops.noop_metadata(accumulation_strategy="atomic_add")

    assert metadata["primitive"] == "noop_metadata"
    assert metadata["accumulation_strategy"] == "atomic_add"
    ops.validate_metadata(metadata)
