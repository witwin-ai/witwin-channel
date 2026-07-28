"""AD-2 layer 1: RayD reflection EPC paths geometry JVP/VJP versus central FD.

Exercises the new fixed-winner C-ABI directly (channel bridge facades
only): d(hit points, emitted unit normals, path length) / d(vertices, source,
receiver) for the direct-plane EPC forward, at a frozen winner sequence, plus
jvp-vs-vjp inner-product duality and the face-normal table companions used by
the transmission seam. Vertex probes rebuild the native scene per FD
evaluation, exactly like the solver-side FD oracles do.
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import (
    central_difference_directional,
    central_difference_gradient,
    relative_error,
)
from tests.ad._tolerances import (
    ABS_TOL,
    FD_STEP_GEOMETRY,
    FD_STEP_POSITION,
    REL_TOL_GENERAL,
    REL_TOL_PATH,
)
from witwin.core import Mesh, Scene, Structure
from witwin.channel.kernels import geometry as ops
from witwin.channel.kernels import topology as topology_kernels
from witwin.channel.scene import compile as compile_scene
from witwin.core import PhysicalMaterial

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for RayD geometry AD"
)

_WALL_VERTICES = (
    (2.5, -3.0, -1.0),
    (2.5, 3.0, -1.0),
    (2.5, -3.0, 2.4),
    (2.5, 3.0, 2.4),
)
_WALL_FACES = ((0, 1, 2), (1, 3, 2))

_CORRIDOR_VERTICES = _WALL_VERTICES + (
    (-2.5, -3.0, -1.0),
    (-2.5, 3.0, -1.0),
    (-2.5, -3.0, 2.4),
    (-2.5, 3.0, 2.4),
)
_CORRIDOR_FACES = _WALL_FACES + ((4, 5, 6), (5, 7, 6))


def _source_linked_rayd_available() -> bool:
    from witwin.channel.deployment import build_info

    try:
        return build_info()["rayd_integration"] == "source-linked"
    except ModuleNotFoundError:
        return False


def _build_rayd_scene(vertices: torch.Tensor, faces) -> object:
    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")
    scene = Scene(
        structures=[
            Structure(
                geometry=Mesh(
                    vertices.detach().cpu().to(torch.float32),
                    torch.tensor(faces, dtype=torch.int32),
                    recenter=False,
                    fill_mode="surface",
                    topology_diagnostics=False,
                ),
                material=PhysicalMaterial(eps_r=2.0),
            )
        ],
        endpoints=[],
    )
    return compile_scene(scene, reference_frequency_hz=3.5e9).rayd


def _plane_inputs(
    rayd: object, sequence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct-plane arrays exactly as the solver seam builds them."""

    records = rayd.edge_records()
    tri_a = topology_kernels.deterministic_face_anchor_points(
        records.vertices.contiguous(), records.faces.contiguous()
    )
    normals = ops.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    face_id = sequence.to(dtype=torch.int64)
    return tri_a[face_id].contiguous(), normals[face_id].contiguous()


def _single_group(rayd: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records = rayd.edge_records()
    face_count = int(records.faces.shape[0])
    device = records.faces.device
    return (
        torch.zeros(face_count, device=device, dtype=torch.int32),
        torch.tensor([face_count], device=device, dtype=torch.int32),
        torch.arange(face_count, device=device, dtype=torch.int32),
    )


def _wall_groups(rayd: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One coplanar group per wall of the two-wall corridor fixture."""

    device = rayd.edge_records().faces.device
    group_id = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
    group_size = torch.tensor([2, 2], device=device, dtype=torch.int32)
    members = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    return group_id, group_size, members


def _single_bounce_case() -> dict[str, object]:
    rayd = _build_rayd_scene(
        torch.tensor(_WALL_VERTICES, dtype=torch.float32), _WALL_FACES
    )
    source = torch.tensor([[0.0, -1.0, 0.5]], dtype=torch.float32, device="cuda")
    receiver = torch.tensor([[0.1, 1.2, 0.4]], dtype=torch.float32, device="cuda")
    sequence = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    groups = _single_group(rayd)
    return {
        "rayd": rayd,
        "faces": _WALL_FACES,
        "base_vertices": torch.tensor(_WALL_VERTICES, dtype=torch.float32),
        "source": source,
        "receiver": receiver,
        "sequence": sequence,
        "groups": groups,
        "depth": 1,
    }


def _two_bounce_case() -> dict[str, object]:
    rayd = _build_rayd_scene(
        torch.tensor(_CORRIDOR_VERTICES, dtype=torch.float32), _CORRIDOR_FACES
    )
    source = torch.tensor([[0.3, -1.0, 0.5]], dtype=torch.float32, device="cuda")
    receiver = torch.tensor([[-0.4, 1.0, 0.6]], dtype=torch.float32, device="cuda")
    sequence = torch.tensor([[0, 2]], dtype=torch.int32, device="cuda")
    groups = _wall_groups(rayd)
    return {
        "rayd": rayd,
        "faces": _CORRIDOR_FACES,
        "base_vertices": torch.tensor(_CORRIDOR_VERTICES, dtype=torch.float32),
        "source": source,
        "receiver": receiver,
        "sequence": sequence,
        "groups": groups,
        "depth": 2,
    }


def _epc_forward(case: dict[str, object], rayd=None, source=None, receiver=None):
    rayd = case["rayd"] if rayd is None else rayd
    source = case["source"] if source is None else source
    receiver = case["receiver"] if receiver is None else receiver
    plane_points, plane_normals = _plane_inputs(rayd, case["sequence"])
    group_id, group_size, group_members = case["groups"]
    out = ops.rayd_reflection_epc_paths_forward(
        rayd.require_resource(),
        source.contiguous(),
        receiver.contiguous(),
        None,
        case["sequence"],
        plane_points,
        plane_normals,
        group_id,
        group_size,
        group_members,
        case["depth"],
        1,
    )
    assert bool(out[0].all())
    return out, plane_points, plane_normals


def _frozen_winner(case: dict[str, object]):
    out, plane_points, plane_normals = _epc_forward(case)
    valid = out[0]
    bounce_count = torch.full(
        (int(valid.shape[0]),), case["depth"], device="cuda", dtype=torch.int32
    )
    return out, plane_points, plane_normals, valid, bounce_count


def _loss_weights(case: dict[str, object], seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = int(case["source"].shape[0])
    depth = case["depth"]
    w_hits = torch.randn(rows, depth, 3, generator=generator).to("cuda")
    w_normals = torch.randn(rows, depth, 3, generator=generator).to("cuda")
    w_length = torch.randn(rows, generator=generator).to("cuda")
    return w_hits, w_normals, w_length


def _weighted_loss(out, weights) -> torch.Tensor:
    w_hits, w_normals, w_length = weights
    loss = (w_hits.double() * out[4].double()).sum()
    loss = loss + (w_normals.double() * out[5].double()).sum()
    return loss + (w_length.double() * out[1].double()).sum()


@pytest.mark.parametrize("case_builder", (_single_bounce_case, _two_bounce_case))
def test_epc_paths_plane_contract_matches_scene_tables(case_builder):
    """Gate from the spec: the plane channel passes for prim P must be
    the plane of that triangle in RayD's own scene tables (anchor = v0,
    normal = normalize(cross(v1 - v0, v2 - v0))), and the emitted hits must
    lie on it."""

    case = case_builder()
    records = case["rayd"].edge_records()
    faces = records.faces.to(dtype=torch.int64)
    vertices = records.vertices
    face_id = case["sequence"].to(dtype=torch.int64)

    plane_points, plane_normals = _plane_inputs(case["rayd"], case["sequence"])
    v0 = vertices[faces[face_id][..., 0]]
    v1 = vertices[faces[face_id][..., 1]]
    v2 = vertices[faces[face_id][..., 2]]
    expected_normal = torch.cross(v1 - v0, v2 - v0, dim=-1)
    expected_normal = expected_normal / expected_normal.norm(dim=-1, keepdim=True)
    assert torch.allclose(plane_points, v0, atol=1.0e-6)
    assert torch.allclose(plane_normals, expected_normal, atol=1.0e-5)

    out, _, _ = _epc_forward(case)
    offset = ((out[4] - plane_points) * plane_normals).sum(-1)
    assert float(offset.abs().max()) <= 1.0e-4


@pytest.mark.parametrize("case_builder", (_single_bounce_case, _two_bounce_case))
def test_epc_paths_backward_matches_fd_wrt_endpoints(case_builder):
    case = case_builder()
    weights = _loss_weights(case, seed=101)
    _out, plane_points, plane_normals, valid, bounce_count = _frozen_winner(case)

    grads = ops.rayd_reflection_epc_paths_backward(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        grad_points=weights[0],
        grad_normals=weights[1],
        grad_path_length=weights[2],
        need_grad_source=True,
        need_grad_receiver=True,
    )

    def source_loss(x: torch.Tensor) -> torch.Tensor:
        out, _, _ = _epc_forward(case, source=x)
        return _weighted_loss(out, weights)

    def receiver_loss(x: torch.Tensor) -> torch.Tensor:
        out, _, _ = _epc_forward(case, receiver=x)
        return _weighted_loss(out, weights)

    fd_source = central_difference_gradient(
        source_loss, case["source"], FD_STEP_POSITION
    )
    fd_receiver = central_difference_gradient(
        receiver_loss, case["receiver"], FD_STEP_POSITION
    )
    assert relative_error(grads[1], fd_source, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    assert relative_error(grads[2], fd_receiver, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("case_builder", (_single_bounce_case, _two_bounce_case))
def test_epc_paths_backward_matches_fd_wrt_vertices(case_builder):
    case = case_builder()
    weights = _loss_weights(case, seed=103)
    _out, plane_points, plane_normals, valid, bounce_count = _frozen_winner(case)

    grads = ops.rayd_reflection_epc_paths_backward(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        grad_points=weights[0],
        grad_normals=weights[1],
        grad_path_length=weights[2],
        need_grad_vertices=True,
    )

    def rebuild_loss(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt = _build_rayd_scene(perturbed_vertices, case["faces"])
        out, _, _ = _epc_forward(case, rayd=rebuilt)
        return _weighted_loss(out, weights)

    generator = torch.Generator(device="cpu").manual_seed(107)
    for _ in range(3):
        direction = torch.randn(case["base_vertices"].shape, generator=generator).to(
            "cuda"
        )
        fd_value = central_difference_directional(
            rebuild_loss, case["base_vertices"].cuda(), direction, FD_STEP_GEOMETRY
        )
        ad_value = (grads[0].double() * direction.double()).sum().cpu()
        assert relative_error(ad_value, fd_value, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_epc_paths_jvp_matches_fd_wrt_endpoints():
    case = _single_bounce_case()
    _out, plane_points, plane_normals, valid, bounce_count = _frozen_winner(case)

    generator = torch.Generator(device="cpu").manual_seed(109)
    v_source = torch.randn(case["source"].shape, generator=generator).to("cuda")
    v_receiver = torch.randn(case["receiver"].shape, generator=generator).to("cuda")
    tangents = ops.rayd_reflection_epc_paths_jvp(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        tangent_source=v_source,
        tangent_receiver=v_receiver,
    )

    packed = torch.cat([case["source"], case["receiver"]], dim=0)
    direction = torch.cat([v_source, v_receiver], dim=0)

    def forward_outputs(x: torch.Tensor, index: int) -> torch.Tensor:
        out, _, _ = _epc_forward(case, source=x[:1], receiver=x[1:])
        return out[index]

    fd_hits = central_difference_directional(
        lambda x: forward_outputs(x, 4), packed, direction, FD_STEP_POSITION
    )
    fd_normals = central_difference_directional(
        lambda x: forward_outputs(x, 5), packed, direction, FD_STEP_POSITION
    )
    fd_length = central_difference_directional(
        lambda x: forward_outputs(x, 1), packed, direction, FD_STEP_POSITION
    )
    assert relative_error(tangents[0], fd_hits, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    # The plane normals do not depend on the endpoints: exact zero tangents.
    assert float(tangents[1].abs().max()) == 0.0
    assert float(fd_normals.abs().max()) <= 1.0e-4
    assert relative_error(tangents[2], fd_length, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_epc_paths_jvp_matches_fd_wrt_vertices():
    case = _single_bounce_case()
    _out, plane_points, plane_normals, valid, bounce_count = _frozen_winner(case)

    generator = torch.Generator(device="cpu").manual_seed(113)
    direction = torch.randn(case["base_vertices"].shape, generator=generator).to("cuda")
    tangents = ops.rayd_reflection_epc_paths_jvp(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        tangent_vertices=direction,
    )

    def rebuild_outputs(perturbed_vertices: torch.Tensor, index: int) -> torch.Tensor:
        rebuilt = _build_rayd_scene(perturbed_vertices, case["faces"])
        out, _, _ = _epc_forward(case, rayd=rebuilt)
        return out[index]

    base = case["base_vertices"].cuda()
    fd_hits = central_difference_directional(
        lambda x: rebuild_outputs(x, 4), base, direction, FD_STEP_GEOMETRY
    )
    fd_normals = central_difference_directional(
        lambda x: rebuild_outputs(x, 5), base, direction, FD_STEP_GEOMETRY
    )
    fd_length = central_difference_directional(
        lambda x: rebuild_outputs(x, 1), base, direction, FD_STEP_GEOMETRY
    )
    assert relative_error(tangents[0], fd_hits, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    assert relative_error(tangents[1], fd_normals, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    assert relative_error(tangents[2], fd_length, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("case_builder", (_single_bounce_case, _two_bounce_case))
def test_epc_paths_jvp_vjp_inner_product_duality(case_builder):
    case = case_builder()
    _out, plane_points, plane_normals, valid, bounce_count = _frozen_winner(case)

    generator = torch.Generator(device="cpu").manual_seed(127)
    rows = int(case["source"].shape[0])
    depth = case["depth"]
    u_hits = torch.randn(rows, depth, 3, generator=generator).to("cuda")
    u_normals = torch.randn(rows, depth, 3, generator=generator).to("cuda")
    u_length = torch.randn(rows, generator=generator).to("cuda")
    v_vertices = torch.randn(case["base_vertices"].shape, generator=generator).to(
        "cuda"
    )
    v_source = torch.randn(case["source"].shape, generator=generator).to("cuda")
    v_receiver = torch.randn(case["receiver"].shape, generator=generator).to("cuda")

    tangents = ops.rayd_reflection_epc_paths_jvp(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        tangent_vertices=v_vertices,
        tangent_source=v_source,
        tangent_receiver=v_receiver,
    )
    lhs = (tangents[0].double() * u_hits.double()).sum()
    lhs = lhs + (tangents[1].double() * u_normals.double()).sum()
    lhs = lhs + (tangents[2].double() * u_length.double()).sum()

    grads = ops.rayd_reflection_epc_paths_backward(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        valid,
        bounce_count,
        grad_points=u_hits,
        grad_normals=u_normals,
        grad_path_length=u_length,
        need_grad_vertices=True,
        need_grad_source=True,
        need_grad_receiver=True,
    )
    rhs = (grads[0].double() * v_vertices.double()).sum()
    rhs = rhs + (grads[1].double() * v_source.double()).sum()
    rhs = rhs + (grads[2].double() * v_receiver.double()).sum()

    assert relative_error(lhs.cpu(), rhs.cpu(), abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_epc_paths_ad_function_routes_reverse_and_forward_mode():
    """The thin autograd.Function: reverse-mode grads reach the leaves, the
    frozen outputs stay detached, and torch.func.jvp matches the native jvp."""

    case = _single_bounce_case()
    records = case["rayd"].edge_records()
    plane_points, plane_normals = _plane_inputs(case["rayd"], case["sequence"])
    group_id, group_size, group_members = case["groups"]

    vertices_ad = records.vertices.detach().clone().requires_grad_(True)
    source_ad = case["source"].clone().requires_grad_(True)
    receiver_ad = case["receiver"].clone().requires_grad_(True)
    out = ops.rayd_reflection_epc_paths_ad(
        case["rayd"],
        vertices_ad,
        source_ad,
        receiver_ad,
        case["sequence"],
        plane_points,
        plane_normals,
        group_id,
        group_size,
        group_members,
        case["depth"],
        1,
    )
    assert bool(out["valid"].all())
    assert out["hit_positions"].requires_grad
    assert out["normals"].requires_grad
    assert out["path_length"].requires_grad
    for name in ("valid", "resolved_prim_ids", "surface_group_ids"):
        assert not out[name].requires_grad
        assert out[name].grad_fn is None

    weights = _loss_weights(case, seed=131)
    loss = (weights[0] * out["hit_positions"]).sum()
    loss = loss + (weights[1] * out["normals"]).sum()
    loss = loss + (weights[2] * out["path_length"]).sum()
    loss.backward()
    assert source_ad.grad is not None and float(source_ad.grad.abs().max()) > 0.0
    assert receiver_ad.grad is not None and float(receiver_ad.grad.abs().max()) > 0.0
    assert vertices_ad.grad is not None and float(vertices_ad.grad.abs().max()) > 0.0

    # The Function VJP must be the native backward exactly.
    _out, pp, pn, valid, bounce_count = _frozen_winner(case)
    grads = ops.rayd_reflection_epc_paths_backward(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        pp,
        pn,
        valid,
        bounce_count,
        grad_points=weights[0],
        grad_normals=weights[1],
        grad_path_length=weights[2],
        need_grad_vertices=True,
        need_grad_source=True,
        need_grad_receiver=True,
    )
    assert torch.allclose(vertices_ad.grad, grads[0])
    assert torch.allclose(source_ad.grad, grads[1])
    assert torch.allclose(receiver_ad.grad, grads[2])

    generator = torch.Generator(device="cpu").manual_seed(137)
    v_source = torch.randn(case["source"].shape, generator=generator).to("cuda")
    expected = ops.rayd_reflection_epc_paths_jvp(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        pp,
        pn,
        valid,
        bounce_count,
        tangent_source=v_source,
    )

    def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = ops.rayd_reflection_epc_paths_ad(
            case["rayd"],
            records.vertices,
            x,
            case["receiver"],
            case["sequence"],
            plane_points,
            plane_normals,
            group_id,
            group_size,
            group_members,
            case["depth"],
            1,
        )
        return result["hit_positions"], result["path_length"]

    _primals, tangents = torch.func.jvp(f, (case["source"],), (v_source,))
    assert relative_error(tangents[0], expected[0], abs_floor=ABS_TOL) <= REL_TOL_PATH
    assert relative_error(tangents[1], expected[2], abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_face_normals_companions_match_fd_and_duality():
    case = _single_bounce_case()
    records = case["rayd"].edge_records()
    face_count = int(records.faces.shape[0])

    generator = torch.Generator(device="cpu").manual_seed(139)
    u_normals = torch.randn(face_count, 3, generator=generator).to("cuda")
    v_vertices = torch.randn(case["base_vertices"].shape, generator=generator).to(
        "cuda"
    )

    grad_vertices = ops.rayd_scene_face_normals_backward(case["rayd"], u_normals)
    tangent_normals = ops.rayd_scene_face_normals_jvp(case["rayd"], v_vertices)
    lhs = (tangent_normals.double() * u_normals.double()).sum()
    rhs = (grad_vertices.double() * v_vertices.double()).sum()
    assert relative_error(lhs.cpu(), rhs.cpu(), abs_floor=ABS_TOL) <= REL_TOL_PATH

    def rebuild_table(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt = _build_rayd_scene(perturbed_vertices, case["faces"])
        rebuilt_records = rebuilt.edge_records()
        return ops.deterministic_normalize_vec3(
            rebuilt_records.face_normals.contiguous(), eps=1.0e-6
        )

    fd_normals = central_difference_directional(
        rebuild_table, case["base_vertices"].cuda(), v_vertices, FD_STEP_GEOMETRY
    )
    assert (
        relative_error(tangent_normals, fd_normals, abs_floor=ABS_TOL)
        <= REL_TOL_GENERAL
    )


def test_face_normals_ad_function_routes_vertex_gradients():
    case = _single_bounce_case()
    records = case["rayd"].edge_records()

    vertices_ad = records.vertices.detach().clone().requires_grad_(True)
    table = ops.rayd_face_normals_ad(
        case["rayd"], vertices_ad, records.face_normals.contiguous()
    )
    assert table.requires_grad
    expected = ops.deterministic_normalize_vec3(
        records.face_normals.contiguous(), eps=1.0e-6
    )
    assert torch.allclose(table.detach(), expected)

    generator = torch.Generator(device="cpu").manual_seed(149)
    u_normals = torch.randn(table.shape, generator=generator).to("cuda")
    (table * u_normals).sum().backward()
    assert vertices_ad.grad is not None
    native = ops.rayd_scene_face_normals_backward(case["rayd"], u_normals)
    assert torch.allclose(vertices_ad.grad, native)

    v_vertices = torch.randn(vertices_ad.shape, generator=generator).to("cuda")
    expected_tangent = ops.rayd_scene_face_normals_jvp(case["rayd"], v_vertices)

    def f(x: torch.Tensor) -> torch.Tensor:
        return ops.rayd_face_normals_ad(
            case["rayd"], x, records.face_normals.contiguous()
        )

    _primal, tangent = torch.func.jvp(
        f, (records.vertices.detach().clone(),), (v_vertices,)
    )
    assert relative_error(tangent, expected_tangent, abs_floor=ABS_TOL) <= REL_TOL_PATH


# Healthy triangle (faces row 0) plus a sliver triangle (faces row 1) whose
# raw cross product has |cross| = 1e-8, well below the face-normal table's
# 1e-6 normalize clamp. The sliver's perturbed coordinates (base 0 or 1e-8)
# stay exactly representable in float32, so the scene-rebuild FD is clean.
_SLIVER_VERTICES = (
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.3),
    (0.0, 2.0, 0.2),
    (5.0, 0.0, 0.0),
    (6.0, 0.0, 0.0),
    (7.0, 1.0e-8, 0.0),
)
_SLIVER_FACES = ((0, 1, 2), (3, 4, 5))
# FD step for sliver perturbations: keeps |cross| <= ~1.1e-7 on both sides of
# the difference, so the primal never leaves the clamped branch.
_SLIVER_FD_STEP = 1.0e-7


def test_face_normals_companions_follow_sliver_clamp_branch():
    """Below the table's 1e-6 normalize clamp the primal is the constant
    scale raw / 1e-6, so its exact derivative is that constant times the
    identity: no projection (which would drop the radial component) and no
    1/|raw| scale (which would blow up as the face degenerates). Adjoint and
    tangent are FD-anchored against the true clamped primal via scene
    rebuild; the healthy face in the same scene keeps its projection-branch
    derivative."""

    base = torch.tensor(_SLIVER_VERTICES, dtype=torch.float32)
    rayd = _build_rayd_scene(base, _SLIVER_FACES)
    records = rayd.edge_records()
    raw = records.face_normals
    # The fixture must actually sit below the clamp, or this test is vacuous.
    assert float(raw[1].norm()) < 1.0e-6
    assert float(raw[0].norm()) > 1.0e-6

    generator = torch.Generator(device="cpu").manual_seed(157)
    w = torch.randn(2, 3, generator=generator).to("cuda")

    vertices_ad = records.vertices.detach().clone().requires_grad_(True)
    table = ops.rayd_face_normals_ad(rayd, vertices_ad, raw.contiguous())
    (w * table).sum().backward()
    assert vertices_ad.grad is not None

    def rebuild_loss(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt = _build_rayd_scene(perturbed_vertices, _SLIVER_FACES)
        rebuilt_table = ops.deterministic_normalize_vec3(
            rebuilt.edge_records().face_normals.contiguous(), eps=1.0e-6
        )
        return (w.double().cpu() * rebuilt_table.double().cpu()).sum()

    # Radial direction: moves the raw cross along its own axis, the component
    # an unclamped projection adjoint drops entirely.
    radial = torch.zeros(6, 3)
    radial[5, 1] = 1.0
    # Tangential direction: stays in the projection subspace, where an
    # unclamped adjoint would scale by 1/|raw| = 1e8 instead of 1/eps = 1e6.
    tangential = torch.zeros(6, 3)
    tangential[5, 2] = 1.0
    # Healthy direction: only the healthy triangle's vertices move.
    healthy = torch.zeros(6, 3)
    healthy[:3] = torch.randn(3, 3, generator=generator)

    for direction, step in (
        (radial, _SLIVER_FD_STEP),
        (tangential, _SLIVER_FD_STEP),
        (healthy, FD_STEP_GEOMETRY),
    ):
        direction = direction.to("cuda")
        fd_value = central_difference_directional(
            rebuild_loss, base.cuda(), direction, step
        )
        vjp_value = (vertices_ad.grad.double() * direction.double()).sum().cpu()
        tangent = ops.rayd_scene_face_normals_jvp(rayd, direction.contiguous())
        jvp_value = (w.double().cpu() * tangent.double().cpu()).sum()
        assert float(fd_value.abs()) > 0.0
        assert relative_error(vjp_value, fd_value, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
        assert relative_error(jvp_value, fd_value, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_epc_paths_backward_skips_invalid_rows():
    """A row the discovery rejected must contribute exactly zero gradient."""

    case = _single_bounce_case()
    _out, plane_points, plane_normals, valid, bounce_count = _frozen_winner(case)
    invalid = torch.zeros_like(valid)

    weights = _loss_weights(case, seed=151)
    grads = ops.rayd_reflection_epc_paths_backward(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        invalid,
        bounce_count,
        grad_points=weights[0],
        grad_normals=weights[1],
        grad_path_length=weights[2],
        need_grad_vertices=True,
        need_grad_source=True,
        need_grad_receiver=True,
    )
    tangents = ops.rayd_reflection_epc_paths_jvp(
        case["rayd"],
        case["source"],
        case["receiver"],
        case["sequence"],
        plane_points,
        plane_normals,
        invalid,
        bounce_count,
        tangent_source=torch.ones_like(case["source"]),
        tangent_receiver=torch.ones_like(case["receiver"]),
    )
    for tensor in (*grads, *tangents):
        assert float(tensor.abs().max()) == 0.0


def test_epc_paths_facades_reject_mismatched_batches():
    case = _single_bounce_case()
    _out, plane_points, plane_normals, valid, bounce_count = _frozen_winner(case)

    source2 = case["source"].repeat(2, 1).contiguous()
    receiver2 = case["receiver"].repeat(2, 1).contiguous()
    with pytest.raises(ValueError):
        ops.rayd_reflection_epc_paths_backward(
            case["rayd"],
            source2,
            receiver2,
            case["sequence"],
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            grad_path_length=torch.ones(2, device="cuda"),
            need_grad_source=True,
        )
    with pytest.raises(ValueError):
        ops.rayd_reflection_epc_paths_jvp(
            case["rayd"],
            source2,
            receiver2,
            case["sequence"],
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            tangent_source=torch.zeros_like(source2),
        )
    # A vertex tangent that is not the scene global vertex table is rejected
    # by the native entry before any kernel launch.
    bad_vertices = torch.zeros(
        (int(case["base_vertices"].shape[0]) + 3, 3),
        dtype=torch.float32,
        device="cuda",
    )
    with pytest.raises(RuntimeError):
        ops.rayd_reflection_epc_paths_jvp(
            case["rayd"],
            case["source"],
            case["receiver"],
            case["sequence"],
            plane_points,
            plane_normals,
            valid,
            bounce_count,
            tangent_vertices=bad_vertices,
        )
    with pytest.raises(RuntimeError):
        ops.rayd_scene_face_normals_jvp(case["rayd"], bad_vertices)
