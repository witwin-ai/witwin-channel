from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor


def bdpt_sample_directions(
    count: int, reference: torch.Tensor, *, seed: int
) -> torch.Tensor:
    if count < 0:
        raise ValueError("count must be non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    validate_cuda_tensor("reference", reference, dtype=torch.float32, ndim=2)
    directions = _required_native_op("bdpt_sample_directions")(
        int(count), reference, int(seed)
    )
    if not isinstance(directions, torch.Tensor):
        raise TypeError("_channel_native.bdpt_sample_directions must return a tensor")
    validate_cuda_tensor(
        "directions", directions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return directions


def bdpt_reflection_launch_inputs(
    tx_positions: torch.Tensor,
    *,
    tx_index: int,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    validate_cuda_tensor(
        "tx_positions", tx_positions, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    exported = _required_native_op("bdpt_reflection_launch_inputs")(
        tx_positions, int(tx_index), int(sample_count)
    )
    if not isinstance(exported, dict):
        raise TypeError(
            "_channel_native.bdpt_reflection_launch_inputs must return a dict"
        )
    validate_cuda_tensor(
        "ray_o", exported["ray_o"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("ray_tmax", exported["ray_tmax"], dtype=torch.float32, ndim=1)
    validate_cuda_tensor("active", exported["active"], dtype=torch.bool, ndim=1)
    validate_cuda_tensor(
        "tx_pol", exported["tx_pol"], dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    validate_cuda_tensor("tx_id", exported["tx_id"], dtype=torch.int32, ndim=1)
    validate_cuda_tensor(
        "light_seed", exported["light_seed"], dtype=torch.int64, ndim=1
    )
    return exported


def bdpt_pack_vec3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    validate_cuda_tensor("x", x, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("y", y, dtype=torch.float32, ndim=1)
    validate_cuda_tensor("z", z, dtype=torch.float32, ndim=1)
    packed = _required_native_op("bdpt_pack_vec3")(x, y, z)
    if not isinstance(packed, torch.Tensor):
        raise TypeError("_channel_native.bdpt_pack_vec3 must return a tensor")
    validate_cuda_tensor(
        "packed", packed, dtype=torch.float32, ndim=2, trailing_shape=(3,)
    )
    return packed
