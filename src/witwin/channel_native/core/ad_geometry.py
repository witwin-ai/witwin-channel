"""Fixed-winner differentiable hit geometry for the plan 07 AD-2 seam.

Topology discovery stays native and detached: RayDN finds the winner (face
sequence, validity, visibility) and channel_native freezes it. What this
module adds is the *continuous* half of the fixed-winner contract: it
re-evaluates the closed-form construction that produced the winner's hit
points, from the live scene tensors (mesh vertices, transmitter and receiver
positions), so ``interaction_positions`` / ``interaction_normals`` reach the
differentiable field kernels on the torch graph.

Why re-evaluate instead of reusing RayD's kernels: RayD's C-ABI exposes a
differentiable ``path_length`` for the reflection EPC tape but no derivative
of the interaction points themselves, and the Fresnel amplitudes depend on
the incidence angles, which are *not* stationary (only the total path length
is, by Fermat). The image-source construction below is geometry, not EM
physics, and every reconstruction is checked against the native discovery
output by ``assert_geometry_parity`` so the two can never silently drift.
"""

from __future__ import annotations

import torch

_PARITY_ATOL = 1.0e-3
_NORMAL_ATOL = 1.0e-3
_DENOM_EPS = 1.0e-12


def _normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)


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
    assert_geometry_parity("vertices", vertices, native)
    return vertices


def assert_geometry_parity(
    name: str,
    reconstructed: torch.Tensor,
    native: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> None:
    """Fail loudly when the torch reconstruction leaves the native winner.

    The reconstruction is only legitimate while it reproduces what the native
    discovery actually found; a drift means the gradient would describe a
    different path than the forward did. ``mask`` restricts the comparison to
    the slots the field kernels actually consume.
    """

    if not native.numel():
        return
    difference = (reconstructed.detach() - native).abs()
    if mask is not None:
        difference = difference * mask.to(dtype=difference.dtype)
    deviation = difference.max()
    if float(deviation) > _PARITY_ATOL:
        raise RuntimeError(
            f"differentiable {name} deviate from the native topology by "
            f"{float(deviation):.3e} m (limit {_PARITY_ATOL:.1e}); the "
            "fixed-winner reconstruction no longer describes the discovered "
            "path"
        )


def transmitter_positions_ad(
    scene: object, native: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Live transmitter positions (the native builder flattens to host floats)."""

    if not scene.transmitters:
        return native
    positions = torch.stack(
        [
            transmitter.position.to(device=device, dtype=torch.float32)
            for transmitter in scene.transmitters
        ],
        dim=0,
    )
    assert_geometry_parity("transmitter positions", positions, native)
    return positions


def receiver_positions_ad(
    scene: object, native: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Live receiver positions for point receivers.

    Grid receivers are generated natively from origin/axes/spacing and stay
    detached: a grid exposes no per-receiver position tensor for a user to
    mark requires_grad, so nothing is silently zeroed here.
    """

    from .objects import ReceiverPoint

    if not scene.receivers or not all(
        isinstance(receiver, ReceiverPoint) for receiver in scene.receivers
    ):
        return native
    positions = torch.stack(
        [
            receiver.position.to(device=device, dtype=torch.float32)
            for receiver in scene.receivers
        ],
        dim=0,
    )
    assert_geometry_parity("receiver positions", positions, native)
    return positions


def assert_plane_parity(
    reconstructed: torch.Tensor,
    native: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> None:
    """Same-plane check for reconstructed normals, insensitive to the sign.

    The field kernels flip an interaction normal against the incident ray, so
    only the plane the discovery hit is a contract; the sign RayDN happened to
    emit is not.
    """

    if not native.numel():
        return
    alignment = (reconstructed.detach() * native).sum(-1).abs()
    deviation = (alignment - 1.0).abs()
    if mask is not None:
        deviation = deviation * mask.to(dtype=deviation.dtype)
    if float(deviation.max()) > _NORMAL_ATOL:
        raise RuntimeError(
            "differentiable interaction normals no longer lie in the native "
            f"interaction plane (max |cos| deviation {float(deviation.max()):.3e})"
        )


def face_planes(
    vertices: torch.Tensor, faces: torch.Tensor, face_id: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plane anchor and unit normal of the given faces, differentiable in vertices."""

    corners = faces.to(dtype=torch.int64)[face_id]
    v0 = vertices[corners[..., 0]]
    v1 = vertices[corners[..., 1]]
    v2 = vertices[corners[..., 2]]
    normal = _normalize(torch.cross(v1 - v0, v2 - v0, dim=-1))
    return v0, normal


def _segment_plane_hit(
    origin: torch.Tensor,
    endpoint: torch.Tensor,
    anchor: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    """Intersection of the origin-endpoint line with the plane.

    The winner is already known to hit the plane, so the denominator is
    bounded away from zero; the epsilon only keeps a degenerate row finite
    instead of poisoning the whole batch with NaN.
    """

    span = endpoint - origin
    denominator = (span * normal).sum(-1, keepdim=True)
    guarded = torch.where(
        denominator.abs() < _DENOM_EPS,
        torch.full_like(denominator, _DENOM_EPS),
        denominator,
    )
    step = ((anchor - origin) * normal).sum(-1, keepdim=True) / guarded
    return origin + step * span


def specular_chain_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_sequence: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image-source reconstruction of a reflection chain's hit points.

    ``face_sequence`` is the frozen winner (rows x depth global face ids).
    Mirrors the source through each plane in order, then walks the images
    back from the receiver, which is exactly the construction the native EPC
    discovery solves. Differentiable in ``vertices``, ``source``, ``target``.
    """

    depth = int(face_sequence.shape[1])
    anchors: list[torch.Tensor] = []
    normals: list[torch.Tensor] = []
    images: list[torch.Tensor] = []
    image = source
    for bounce in range(depth):
        anchor, normal = face_planes(vertices, faces, face_sequence[:, bounce])
        offset = ((image - anchor) * normal).sum(-1, keepdim=True)
        image = image - 2.0 * offset * normal
        anchors.append(anchor)
        normals.append(normal)
        images.append(image)

    positions: list[torch.Tensor] = [torch.empty(0)] * depth
    endpoint = target
    for bounce in range(depth - 1, -1, -1):
        hit = _segment_plane_hit(
            images[bounce], endpoint, anchors[bounce], normals[bounce]
        )
        positions[bounce] = hit
        endpoint = hit
    return torch.stack(positions, dim=1), torch.stack(normals, dim=1)


def straight_chain_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_sequence: torch.Tensor,
    event_valid: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wall-crossing points of a straight transmission path.

    Transmission keeps the LoS ray: every crossed wall contributes the
    intersection of the source-target segment with that face's plane. Invalid
    slots are skipped by the kernel, so they receive a zero cotangent and the
    face id clamp below cannot leak a gradient into face 0.
    """

    width = int(face_sequence.shape[1])
    anchor, normal = face_planes(vertices, faces, face_sequence.clamp_min(0))
    origin = source.unsqueeze(1).expand(-1, width, -1)
    endpoint = target.unsqueeze(1).expand(-1, width, -1)
    hit = _segment_plane_hit(origin, endpoint, anchor, normal)
    return torch.where(event_valid.unsqueeze(-1), hit, origin), normal
