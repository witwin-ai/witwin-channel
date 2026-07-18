"""ISB boundary taper (ADR-017) line-of-sight member facade.

Thin facades over the native ``los_silhouette_clearance`` / ``los_taper_apply``
CUDA ops. These are only ever invoked when the DEFAULT-OFF ``isb_boundary_taper``
switch is on, so the off solve never imports or launches anything here and stays
bit-identical.

``occluder_boxes`` builds the per-structure axis-aligned box table once from the
compiled geometry vertices. That is a compile-time structural reduction over the
scene's handful of structures (not a per-(tx, rx) hot path); the per-pair
clearance physics runs entirely in the native kernel.
"""

from __future__ import annotations

import torch

from witwin.channel_native.runtime.symbols import required_symbol as _required_native_op
from witwin.channel_native.runtime.tensor_contracts import validate_cuda_tensor

# Speed of light (m/s); wavelength = c0 / frequency_hz. Matches the LoS kernel
# and artifacts/isb-taper/common.py (lambda = 0.06 m at 5 GHz).
_C0 = 299792458.0


def occluder_boxes(compiled: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-structure axis-aligned box table (box_min, box_max) or ``None``.

    Reduces the compiled geometry vertices to one [min, max] box per structure
    that owns at least one face. Structures with no faces are dropped. Returns
    ``None`` when the scene has no boxed occluder so the caller keeps the
    fully-lit fast path.
    """

    geometry = compiled.geometry
    # The box reduction is a per-structure amin/amax scatter that must run on the
    # scene device: the compiled geometry tables may live on the host, so pin
    # faces / structure ids / vertices to the vertex device before the scatter
    # (a CPU index against a CUDA accumulator raises otherwise).
    device = geometry.vertices.device
    faces = geometry.faces.to(device=device, dtype=torch.int64)
    face_structure_id = geometry.face_structure_id.to(device=device, dtype=torch.int64)
    if face_structure_id.numel() == 0 or faces.numel() == 0:
        return None
    vertices = geometry.vertices.to(device=device, dtype=torch.float32)
    structure_count = int(face_structure_id.max().item()) + 1
    corner_vertex = faces.reshape(-1)
    corner_structure = face_structure_id.repeat_interleave(3)
    corner_position = vertices[corner_vertex]
    scatter_index = corner_structure.unsqueeze(1).expand(-1, 3)
    box_min = torch.full(
        (structure_count, 3), float("inf"), device=device, dtype=torch.float32
    )
    box_max = torch.full(
        (structure_count, 3), float("-inf"), device=device, dtype=torch.float32
    )
    box_min.scatter_reduce_(
        0, scatter_index, corner_position, reduce="amin", include_self=True
    )
    box_max.scatter_reduce_(
        0, scatter_index, corner_position, reduce="amax", include_self=True
    )
    populated = torch.isfinite(box_min).all(dim=1)
    box_min = box_min[populated].contiguous()
    box_max = box_max[populated].contiguous()
    if box_min.shape[0] == 0:
        return None
    return box_min, box_max


def los_clearance_factor(
    source: torch.Tensor,
    target: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
    *,
    frequency_hz: float,
    width: float,
) -> torch.Tensor:
    """Native per-pair ISB membership factor tau in [0, 1] (ADR-017).

    tau = smoothstep01(0.5 * (c_plane / (width * w_F) + 1)) with c the signed
    clearance of the source->target segment past the nearest occluding box
    silhouette (measured at the occluder), c_plane = c * (d1 + d2) / d1 that
    clearance magnified into the receiver plane by the point-source shadow
    factor, and w_F the grazed-edge Fresnel penumbra. The receiver-plane
    magnification matches the accepted projection's in-plane distance transform
    (artifacts/isb-taper/stage2.py); exact conventions live in the CUDA kernel.
    tau > 0 is the membership predicate; tau < 1 is the amplitude factor.
    """

    validate_cuda_tensor("source", source, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("target", target, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("box_min", box_min, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    validate_cuda_tensor("box_max", box_max, dtype=torch.float32, ndim=2, trailing_shape=(3,))
    if target.shape != source.shape:
        raise ValueError("target must match source")
    if box_max.shape != box_min.shape:
        raise ValueError("box_max must match box_min")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if not (0.0 < width <= 4.0):
        raise ValueError("isb_boundary_taper_width must be in (0, 4]")
    wavelength = _C0 / float(frequency_hz)
    tau = _required_native_op("los_silhouette_clearance")(
        source.contiguous(),
        target.contiguous(),
        box_min.contiguous(),
        box_max.contiguous(),
        float(wavelength),
        float(width),
    )
    if not isinstance(tau, torch.Tensor):
        raise TypeError("_channel_native.los_silhouette_clearance must return a tensor")
    validate_cuda_tensor("tau", tau, dtype=torch.float32, ndim=1)
    return tau


def apply_los_taper(
    field_vector: torch.Tensor,
    coefficient: torch.Tensor,
    path_field: torch.Tensor,
    path_gain: torch.Tensor,
    tau: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Native scale of a LoS field bundle by the per-row factor tau (ADR-017).

    tau multiplies the field amplitude; path_gain (a power) is scaled by tau^2.
    """

    validate_cuda_tensor("tau", tau, dtype=torch.float32, ndim=1)
    out = _required_native_op("los_taper_apply")(
        field_vector.contiguous(),
        coefficient.contiguous(),
        path_field.contiguous(),
        path_gain.contiguous(),
        tau.contiguous(),
    )
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise TypeError("_channel_native.los_taper_apply must return four tensors")
    return {
        "field_vector": out[0],
        "coefficient": out[1],
        "path_field": out[2],
        "path_gain": out[3],
    }
