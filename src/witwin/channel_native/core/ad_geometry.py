"""Scene-leaf plumbing for the plan 07 AD-2 geometry seam.

Topology discovery stays native and detached: RayDN finds the winner (face
sequence, validity, visibility) and channel_native freezes it. Geometry
derivatives under that frozen winner come from RayD's own fixed-winner chain
companions (``ops.raydn_reflection_epc_paths_ad`` for reflection hit
geometry, ``ops.raydn_face_normals_ad`` for the transmission wall normals),
so no hit geometry is ever re-derived on the torch side. What remains here
is pure tensor passing: the live scene tensors (mesh vertices, transmitter
and receiver positions) that anchor the autograd graph the native kernels
route gradients and tangents to.
"""

from __future__ import annotations

import torch


def scene_vertex_table(scene: object, compiled: object) -> torch.Tensor:
    """Live global vertex table matching ``compiled.geometry.vertices``.

    RayDN concatenates structure meshes in scene order, so the live table is
    the concatenation of the structure vertex tensors. Returning the live
    tensors (rather than the native export) is what lets mesh-vertex
    gradients exist at all.
    """

    native = compiled.geometry.vertices
    if not scene.structures:
        return native
    vertices = torch.cat(
        [
            structure.vertices.to(device=native.device, dtype=torch.float32)
            for structure in scene.structures
        ],
        dim=0,
    )
    if vertices.shape != native.shape:
        raise RuntimeError(
            "differentiable vertex table does not match the native scene "
            f"table: {tuple(vertices.shape)} vs {tuple(native.shape)}"
        )
    return vertices


def transmitter_positions_ad(
    scene: object, native: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Live transmitter positions (the native builder flattens to host floats)."""

    if not scene.transmitters:
        return native
    return torch.stack(
        [
            transmitter.position.to(device=device, dtype=torch.float32)
            for transmitter in scene.transmitters
        ],
        dim=0,
    )


def receiver_positions_ad(
    scene: object, native: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Live receiver positions for point receivers.

    Grid receivers are generated natively from origin/axes/spacing and stay
    detached: a grid exposes no per-receiver position tensor for a user to
    mark requires_grad, so nothing is silently zeroed here.
    """

    from witwin.channel_native.scene.models import ReceiverPoint

    if not scene.receivers or not all(
        isinstance(receiver, ReceiverPoint) for receiver in scene.receivers
    ):
        return native
    return torch.stack(
        [
            receiver.position.to(device=device, dtype=torch.float32)
            for receiver in scene.receivers
        ],
        dim=0,
    )
