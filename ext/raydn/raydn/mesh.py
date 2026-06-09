from __future__ import annotations

from dataclasses import dataclass
import torch


def _empty_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _require_tensor(value: torch.Tensor, name: str, dtype: torch.dtype, rank: int, last_dim: int | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be CUDA.")
    if value.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}.")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if last_dim is not None and value.shape[-1] != last_dim:
        raise ValueError(f"{name} last dimension must be {last_dim}.")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _require_transform(value: torch.Tensor, name: str) -> None:
    _require_tensor(value, name, torch.float32, 2, 4)
    if value.shape[0] not in (0, 4):
        raise ValueError(f"{name} must be empty or have shape (4, 4).")


@dataclass
class Mesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    uv: torch.Tensor | None = None
    face_uv: torch.Tensor | None = None
    use_face_normals: bool = False
    edges_enabled: bool = True
    to_world_left: torch.Tensor | None = None
    to_world_right: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_tensor(self.vertices, "vertices", torch.float32, 2, 3)
        _require_tensor(self.faces, "faces", torch.int32, 2, 3)
        if self.uv is not None:
            _require_tensor(self.uv, "uv", torch.float32, 2, 2)
        if self.face_uv is not None:
            _require_tensor(self.face_uv, "face_uv", torch.int32, 2, 3)
        if self.uv is None:
            self.uv = _empty_tensor((0, 2), torch.float32, self.vertices.device)
        if self.face_uv is None:
            self.face_uv = _empty_tensor((0, 3), torch.int32, self.vertices.device)
        if self.to_world_left is None:
            self.to_world_left = _empty_tensor((0, 4), torch.float32, self.vertices.device)
        else:
            _require_transform(self.to_world_left, "to_world_left")
        if self.to_world_right is None:
            self.to_world_right = _empty_tensor((0, 4), torch.float32, self.vertices.device)
        else:
            _require_transform(self.to_world_right, "to_world_right")
