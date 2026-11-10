# Copyright Xingyu Chen.
# Tests shared contract owners.

from __future__ import annotations


import pytest
import torch

from witwin.channel import runtime
from witwin.channel.kernels import montecarlo as paths


def test_validate_cuda_tensor_has_one_canonical_owner():
    assert (
        paths.bdpt_launch_state.__globals__["validate_cuda_tensor"]
        is runtime.validate_cuda_tensor
    )
    assert (
        runtime.validate_cuda_tensor.__module__
        == "witwin.channel.runtime"
    )


def test_validate_cuda_tensor_preserves_exact_error_order_and_text():
    with pytest.raises(TypeError) as error:
        runtime.validate_cuda_tensor(
            "points", object(), dtype=torch.float32, ndim=2
        )
    assert str(error.value) == "points must be a torch.Tensor"

    wrong_dtype = torch.zeros((2, 3), dtype=torch.float64)
    with pytest.raises(TypeError) as error:
        runtime.validate_cuda_tensor(
            "points", wrong_dtype, dtype=torch.float32, ndim=1
        )
    assert str(error.value) == "points must have dtype torch.float32"

    cpu_tensor = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError) as error:
        runtime.validate_cuda_tensor(
            "points", cpu_tensor, dtype=torch.float32, ndim=1
        )
    assert str(error.value) == "points must be a CUDA tensor"
