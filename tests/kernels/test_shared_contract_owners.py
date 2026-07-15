from __future__ import annotations

import ast
import hashlib
import inspect

import pytest
import torch

from witwin.channel_native.core.kernels import metadata, ops
from witwin.channel_native.runtime import tensor_contracts


def _body_hash(function: object) -> str:
    tree = ast.parse(inspect.getsource(function))
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    body = ast.dump(
        ast.Module(body=node.body, type_ignores=[]),
        annotate_fields=True,
        include_attributes=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_validate_cuda_tensor_has_one_canonical_owner_and_preserved_body():
    assert ops.validate_cuda_tensor is tensor_contracts.validate_cuda_tensor
    assert (
        ops.bdpt_launch_state.__globals__["validate_cuda_tensor"]
        is tensor_contracts.validate_cuda_tensor
    )
    assert (
        tensor_contracts.validate_cuda_tensor.__module__
        == "witwin.channel_native.runtime.tensor_contracts"
    )
    assert _body_hash(tensor_contracts.validate_cuda_tensor) == (
        "b3e1c30d4505a02059609e9e3fe3836556ca17c0cb741c39920c6366d9b16af0"
    )


def test_validate_cuda_tensor_preserves_exact_error_order_and_text():
    with pytest.raises(TypeError) as error:
        tensor_contracts.validate_cuda_tensor(
            "points", object(), dtype=torch.float32, ndim=2
        )
    assert str(error.value) == "points must be a torch.Tensor"

    wrong_dtype = torch.zeros((2, 3), dtype=torch.float64)
    with pytest.raises(TypeError) as error:
        tensor_contracts.validate_cuda_tensor(
            "points", wrong_dtype, dtype=torch.float32, ndim=1
        )
    assert str(error.value) == "points must have dtype torch.float32"

    cpu_tensor = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError) as error:
        tensor_contracts.validate_cuda_tensor(
            "points", cpu_tensor, dtype=torch.float32, ndim=1
        )
    assert str(error.value) == "points must be a CUDA tensor"


def test_noop_metadata_has_one_canonical_owner_and_preserved_body():
    assert ops.noop_metadata is metadata.noop_metadata
    assert (
        metadata.noop_metadata.__module__
        == "witwin.channel_native.core.kernels.metadata"
    )
    assert _body_hash(metadata.noop_metadata) == (
        "e38c9bc1703f1d1360e2f971485a447d1aa51700d1399ab8f59f67e75656aed1"
    )
