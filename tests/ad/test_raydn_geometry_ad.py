"""AD-A0: RayDN fixed-winner geometry JVP/VJP versus central finite differences.

Small analytic scenes (a tilted triangle, a quad wall) exercised through the
channel_native C-bridge AD entry points only. Vertex perturbations rebuild the
native scene per FD evaluation; the winner primitive stays fixed by
construction (hit points are far from triangle boundaries).
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
from witwin.channel_native import Scene, Structure
from witwin.channel_native.core.kernels import ops
from witwin.channel_native.core.materials import Dielectric

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for RayDN geometry AD"
)

_TILTED_TRIANGLE_VERTICES = (
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.3),
    (0.0, 2.0, 0.2),
)
_TRIANGLE_FACES = ((0, 1, 2),)

_WALL_VERTICES = (
    (-3.0, -3.0, 0.0),
    (3.0, -3.0, 0.0),
    (3.0, 3.0, 0.0),
    (-3.0, 3.0, 0.0),
)
_WALL_FACES = ((0, 1, 2), (0, 2, 3))


def _source_linked_rayd_available() -> bool:
    from witwin.channel_native.core.kernels.extension import build_info

    try:
        return build_info()["rayd_integration"] == "source-linked"
    except ModuleNotFoundError:
        return False


def _build_raydn_scene(vertices: torch.Tensor, faces) -> tuple[object, torch.Tensor]:
    """Build a single-structure RayDN scene; returns (scene wrapper, global vertices)."""

    if not _source_linked_rayd_available():
        pytest.skip("RayDN native extension is not built")
    scene = Scene(
        structures=[
            Structure(
                vertices=vertices.detach().cpu().to(torch.float32),
                faces=torch.tensor(faces, dtype=torch.int32),
                material=Dielectric(eps_r=2.0),
            )
        ],
        transmitters=[],
        receivers=[],
        frequency=3.5e9,
    )
    raydn = scene.raydn_scene()
    return raydn, raydn.mesh_tensors[0][0]


def _triangle_scene() -> tuple[object, torch.Tensor]:
    return _build_raydn_scene(
        torch.tensor(_TILTED_TRIANGLE_VERTICES, dtype=torch.float32), _TRIANGLE_FACES
    )


def _wall_scene() -> tuple[object, torch.Tensor]:
    return _build_raydn_scene(
        torch.tensor(_WALL_VERTICES, dtype=torch.float32), _WALL_FACES
    )


def _empty_tmax() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32, device="cuda")


def _triangle_rays() -> tuple[torch.Tensor, torch.Tensor]:
    ray_o = torch.tensor(
        [[0.4, 0.5, 1.0], [0.7, 0.3, 1.2]], dtype=torch.float32, device="cuda"
    )
    ray_d = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32, device="cuda"
    )
    return ray_o, ray_d


def _intersect_loss_weights(ray_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(7)
    w_t = torch.randn(ray_count, generator=generator).to("cuda")
    w_p = torch.randn(ray_count, 3, generator=generator).to("cuda")
    return w_t, w_p


def _intersect_fd_loss(
    raydn: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    w_t: torch.Tensor,
    w_p: torch.Tensor,
) -> torch.Tensor:
    hit = ops.bdpt_intersect_forward(
        raydn, ray_o.contiguous(), ray_d, _empty_tmax(), None, flags=7
    )
    loss = (w_t.double() * hit["t"].double()).sum()
    loss = loss + (w_p.double() * hit["p"].double()).sum()
    return loss


def test_intersect_vjp_matches_fd_wrt_ray_origin():
    raydn, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    w_t, w_p = _intersect_loss_weights(ray_o.shape[0])

    ray_o_ad = ray_o.clone().requires_grad_(True)
    out = ops.raydn_intersect_ad(raydn, vertices, ray_o_ad, ray_d, _empty_tmax())
    loss = (w_t * out["t"]).sum() + (w_p * out["p"]).sum()
    loss.backward()
    assert ray_o_ad.grad is not None

    fd_grad = central_difference_gradient(
        lambda x: _intersect_fd_loss(raydn, x, ray_d, w_t, w_p),
        ray_o,
        FD_STEP_POSITION,
    )
    assert (
        relative_error(ray_o_ad.grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


def test_intersect_vjp_matches_fd_wrt_vertices():
    raydn, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    w_t, w_p = _intersect_loss_weights(ray_o.shape[0])

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    out = ops.raydn_intersect_ad(raydn, vertices_ad, ray_o, ray_d, _empty_tmax())
    loss = (w_t * out["t"]).sum() + (w_p * out["p"]).sum()
    loss.backward()
    assert vertices_ad.grad is not None

    def rebuild_loss(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_raydn_scene(perturbed_vertices, _TRIANGLE_FACES)
        return _intersect_fd_loss(rebuilt, ray_o, ray_d, w_t, w_p)

    generator = torch.Generator(device="cpu").manual_seed(11)
    for _ in range(3):
        direction = torch.randn(vertices.shape, generator=generator).to("cuda")
        fd_value = central_difference_directional(
            rebuild_loss, vertices, direction, FD_STEP_GEOMETRY
        )
        ad_value = (
            (vertices_ad.grad.double() * direction.double()).sum().cpu()
        )
        assert relative_error(ad_value, fd_value, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_intersect_jvp_matches_fd_wrt_ray_origin():
    raydn, vertices = _triangle_scene()
    del vertices
    ray_o, ray_d = _triangle_rays()
    hit = ops.bdpt_intersect_forward(raydn, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(13)
    direction = torch.randn(ray_o.shape, generator=generator).to("cuda")
    tangents = ops.raydn_intersect_jvp(
        raydn,
        ray_o,
        ray_d,
        None,
        hit["global_prim_id"],
        hit["barycentric"],
        tangent_ray_o=direction,
    )

    def forward_t(x: torch.Tensor) -> torch.Tensor:
        return ops.bdpt_intersect_forward(
            raydn, x.contiguous(), ray_d, _empty_tmax(), None, flags=7
        )["t"]

    def forward_p(x: torch.Tensor) -> torch.Tensor:
        return ops.bdpt_intersect_forward(
            raydn, x.contiguous(), ray_d, _empty_tmax(), None, flags=7
        )["p"]

    fd_t = central_difference_directional(forward_t, ray_o, direction, FD_STEP_POSITION)
    fd_p = central_difference_directional(forward_p, ray_o, direction, FD_STEP_POSITION)
    assert relative_error(tangents[0], fd_t, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    assert relative_error(tangents[1], fd_p, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_intersect_jvp_matches_fd_wrt_vertices():
    raydn, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.bdpt_intersect_forward(raydn, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(17)
    direction = torch.randn(vertices.shape, generator=generator).to("cuda")
    tangents = ops.raydn_intersect_jvp(
        raydn,
        ray_o,
        ray_d,
        None,
        hit["global_prim_id"],
        hit["barycentric"],
        tangent_vertices=direction,
    )

    def rebuild_t(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_raydn_scene(perturbed_vertices, _TRIANGLE_FACES)
        return ops.bdpt_intersect_forward(
            rebuilt, ray_o, ray_d, _empty_tmax(), None, flags=7
        )["t"]

    def rebuild_p(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_raydn_scene(perturbed_vertices, _TRIANGLE_FACES)
        return ops.bdpt_intersect_forward(
            rebuilt, ray_o, ray_d, _empty_tmax(), None, flags=7
        )["p"]

    fd_t = central_difference_directional(
        rebuild_t, vertices, direction, FD_STEP_GEOMETRY
    )
    fd_p = central_difference_directional(
        rebuild_p, vertices, direction, FD_STEP_GEOMETRY
    )
    assert relative_error(tangents[0], fd_t, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    assert relative_error(tangents[1], fd_p, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_intersect_forward_mode_dual_matches_native_jvp():
    raydn, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.bdpt_intersect_forward(raydn, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(19)
    direction = torch.randn(ray_o.shape, generator=generator).to("cuda")
    expected = ops.raydn_intersect_jvp(
        raydn,
        ray_o,
        ray_d,
        None,
        hit["global_prim_id"],
        hit["barycentric"],
        tangent_ray_o=direction,
    )

    with torch.autograd.forward_ad.dual_level():
        dual_ray_o = torch.autograd.forward_ad.make_dual(ray_o, direction)
        out = ops.raydn_intersect_ad(raydn, vertices, dual_ray_o, ray_d, _empty_tmax())
        _, tangent_t = torch.autograd.forward_ad.unpack_dual(out["t"])
        _, tangent_p = torch.autograd.forward_ad.unpack_dual(out["p"])

    assert tangent_t is not None and tangent_p is not None
    assert relative_error(tangent_t, expected[0], abs_floor=ABS_TOL) <= REL_TOL_PATH
    assert relative_error(tangent_p, expected[1], abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_intersect_jvp_vjp_inner_product_duality():
    raydn, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.bdpt_intersect_forward(raydn, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(23)
    u_t = torch.randn(ray_o.shape[0], generator=generator).to("cuda")
    u_p = torch.randn(ray_o.shape[0], 3, generator=generator).to("cuda")
    v_ray_o = torch.randn(ray_o.shape, generator=generator).to("cuda")
    v_vertices = torch.randn(vertices.shape, generator=generator).to("cuda")

    # <J v, u> from the native JVP.
    tangents = ops.raydn_intersect_jvp(
        raydn,
        ray_o,
        ray_d,
        None,
        hit["global_prim_id"],
        hit["barycentric"],
        tangent_vertices=v_vertices,
        tangent_ray_o=v_ray_o,
    )
    lhs = (tangents[0].double() * u_t.double()).sum()
    lhs = lhs + (tangents[1].double() * u_p.double()).sum()

    # <v, J^T u> from the autograd.Function VJP.
    ray_o_ad = ray_o.clone().requires_grad_(True)
    vertices_ad = vertices.detach().clone().requires_grad_(True)
    out = ops.raydn_intersect_ad(raydn, vertices_ad, ray_o_ad, ray_d, _empty_tmax())
    loss = (u_t * out["t"]).sum() + (u_p * out["p"]).sum()
    loss.backward()
    rhs = (ray_o_ad.grad.double() * v_ray_o.double()).sum()
    rhs = rhs + (vertices_ad.grad.double() * v_vertices.double()).sum()

    assert relative_error(lhs.cpu(), rhs.cpu(), abs_floor=ABS_TOL) <= REL_TOL_PATH


def _wall_reflection_rays() -> tuple[torch.Tensor, torch.Tensor]:
    ray_o = torch.tensor([[0.3, 0.4, 1.0]], dtype=torch.float32, device="cuda")
    ray_d = torch.tensor([[0.2, -0.1, -1.0]], dtype=torch.float32, device="cuda")
    ray_d = ray_d / ray_d.norm(dim=1, keepdim=True)
    return ray_o, ray_d


def _reflection_loss_weights() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(29)
    w_t = torch.randn(1, 1, generator=generator).to("cuda")
    w_img = torch.randn(1, 1, 3, generator=generator).to("cuda")
    return w_t, w_img


def _reflection_fd_loss(
    raydn: object,
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    w_t: torch.Tensor,
    w_img: torch.Tensor,
) -> torch.Tensor:
    forward = ops.raydn_trace_reflections_forward_tape(
        raydn, ray_o.contiguous(), ray_d, _empty_tmax(), None, 1
    )
    loss = (w_t.double() * forward[1].double()).sum()
    loss = loss + (w_img.double() * forward[2].double()).sum()
    return loss


def test_reflection_chain_vjp_matches_fd_wrt_ray_origin():
    raydn, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    w_t, w_img = _reflection_loss_weights()

    ray_o_ad = ray_o.clone().requires_grad_(True)
    out = ops.raydn_trace_reflections_ad(
        raydn, vertices, ray_o_ad, ray_d, _empty_tmax(), None, 1
    )
    assert bool(out["valid"].all())
    loss = (w_t * out["t"]).sum() + (w_img * out["image_sources"]).sum()
    loss.backward()
    assert ray_o_ad.grad is not None

    fd_grad = central_difference_gradient(
        lambda x: _reflection_fd_loss(raydn, x, ray_d, w_t, w_img),
        ray_o,
        FD_STEP_POSITION,
    )
    assert (
        relative_error(ray_o_ad.grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


def test_reflection_chain_vjp_matches_fd_wrt_vertices():
    raydn, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    w_t, w_img = _reflection_loss_weights()

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    out = ops.raydn_trace_reflections_ad(
        raydn, vertices_ad, ray_o, ray_d, _empty_tmax(), None, 1
    )
    assert bool(out["valid"].all())
    loss = (w_t * out["t"]).sum() + (w_img * out["image_sources"]).sum()
    loss.backward()
    assert vertices_ad.grad is not None

    def rebuild_loss(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_raydn_scene(perturbed_vertices, _WALL_FACES)
        return _reflection_fd_loss(rebuilt, ray_o, ray_d, w_t, w_img)

    generator = torch.Generator(device="cpu").manual_seed(31)
    for _ in range(3):
        direction = torch.randn(vertices.shape, generator=generator).to("cuda")
        fd_value = central_difference_directional(
            rebuild_loss, vertices, direction, FD_STEP_GEOMETRY
        )
        ad_value = (vertices_ad.grad.double() * direction.double()).sum().cpu()
        assert relative_error(ad_value, fd_value, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_reflection_chain_jvp_vjp_inner_product_duality():
    raydn, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    forward = ops.raydn_trace_reflections_forward_tape(
        raydn, ray_o, ray_d, _empty_tmax(), None, 1
    )
    _valid, t, image_sources, _prim, tape_prim_id, tape_bary, tape_hits, tape_normals, _active = forward

    generator = torch.Generator(device="cpu").manual_seed(47)
    u_t = torch.randn(t.shape, generator=generator).to("cuda")
    u_img = torch.randn(image_sources.shape, generator=generator).to("cuda")
    v_ray_o = torch.randn(ray_o.shape, generator=generator).to("cuda")
    v_vertices = torch.randn(vertices.shape, generator=generator).to("cuda")

    tangents = ops.raydn_trace_reflections_jvp(
        raydn,
        ray_o,
        ray_d,
        None,
        tape_prim_id,
        tape_bary,
        tape_hits,
        tape_normals,
        image_sources,
        tangent_vertices=v_vertices,
        tangent_ray_o=v_ray_o,
    )
    lhs = (tangents[0].double() * u_t.double()).sum()
    lhs = lhs + (tangents[1].double() * u_img.double()).sum()

    grads = ops.raydn_trace_reflections_backward(
        raydn,
        ray_o,
        ray_d,
        _empty_tmax(),
        None,
        tape_prim_id,
        tape_bary,
        tape_hits,
        tape_normals,
        image_sources,
        grad_t=u_t,
        grad_image_sources=u_img,
    )
    rhs = (grads[0].double() * v_vertices.double()).sum()
    rhs = rhs + (grads[1].double() * v_ray_o.double()).sum()

    assert relative_error(lhs.cpu(), rhs.cpu(), abs_floor=ABS_TOL) <= REL_TOL_PATH


def _epc_endpoints() -> tuple[torch.Tensor, torch.Tensor]:
    # The EPC tape winner is the first primitive on the source -> receiver
    # segment (ray o = source, unnormalized d = receiver - source), so the
    # endpoints must straddle the wall plane.
    source = torch.tensor([[-0.4, 0.2, -1.0]], dtype=torch.float32, device="cuda")
    receiver = torch.tensor([[0.5, -0.1, 0.9]], dtype=torch.float32, device="cuda")
    return source, receiver


def _epc_loss_weights() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(37)
    w_real = torch.randn(1, generator=generator).to("cuda")
    w_imag = torch.randn(1, generator=generator).to("cuda")
    w_len = torch.randn(1, generator=generator).to("cuda")
    return w_real, w_imag, w_len


def _epc_tape(
    raydn: object, source: torch.Tensor, receiver: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Winner tape for the EPC kernels: intersect of ray (source, receiver - source)."""

    hit = ops.bdpt_intersect_forward(
        raydn,
        source.contiguous(),
        (receiver - source).contiguous(),
        _empty_tmax(),
        None,
        flags=7,
    )
    return hit["global_prim_id"], hit["barycentric"], hit["t"]


def _epc_analytic_loss(
    raydn: object,
    source: torch.Tensor,
    receiver: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Primal of the EPC kernels along their tape contract.

    RayD's refl_epc backward/jvp kernels differentiate field_real =
    cos(t) / (1 + t), field_imag = sin(t) / (1 + t) and path_length = t,
    where t is the ray parameter of the winner intersection for the
    unnormalized ray (source, receiver - source).
    """

    _prim, _bary, t = _epc_tape(raydn, source, receiver)
    t = t.double().cpu()
    w_real, w_imag, w_len = (value.double().cpu() for value in weights)
    field_real = torch.cos(t) / (1.0 + t)
    field_imag = torch.sin(t) / (1.0 + t)
    loss = (w_real * field_real).sum()
    loss = loss + (w_imag * field_imag).sum()
    loss = loss + (w_len * t).sum()
    return loss


def test_refl_epc_backward_matches_fd_wrt_source_and_receiver():
    raydn, vertices = _wall_scene()
    del vertices
    source, receiver = _epc_endpoints()
    weights = _epc_loss_weights()
    tape_prim_id, tape_barycentric, tape_t = _epc_tape(raydn, source, receiver)
    assert int(tape_prim_id.min()) >= 0

    w_real, w_imag, w_len = weights
    grads = ops.raydn_refl_epc_backward(
        raydn,
        source,
        receiver,
        None,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        grad_field_real=w_real,
        grad_field_imag=w_imag,
        grad_path_length=w_len,
        need_grad_source=True,
        need_grad_receiver=True,
    )

    fd_source = central_difference_gradient(
        lambda x: _epc_analytic_loss(raydn, x, receiver, weights),
        source,
        FD_STEP_POSITION,
    )
    fd_receiver = central_difference_gradient(
        lambda x: _epc_analytic_loss(raydn, source, x, weights),
        receiver,
        FD_STEP_POSITION,
    )
    assert relative_error(grads[1], fd_source, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    assert relative_error(grads[2], fd_receiver, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_refl_epc_jvp_matches_fd_wrt_source_and_receiver():
    raydn, vertices = _wall_scene()
    del vertices
    source, receiver = _epc_endpoints()
    weights = _epc_loss_weights()
    tape_prim_id, tape_barycentric, tape_t = _epc_tape(raydn, source, receiver)
    assert int(tape_prim_id.min()) >= 0

    generator = torch.Generator(device="cpu").manual_seed(43)
    v_source = torch.randn(source.shape, generator=generator).to("cuda")
    v_receiver = torch.randn(receiver.shape, generator=generator).to("cuda")
    tangents = ops.raydn_refl_epc_jvp(
        raydn,
        source,
        receiver,
        None,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        tangent_source=v_source,
        tangent_receiver=v_receiver,
    )
    w_real, w_imag, w_len = weights
    tangent_loss = (w_real.double().cpu() * tangents[0].double().cpu()).sum()
    tangent_loss = tangent_loss + (w_imag.double().cpu() * tangents[1].double().cpu()).sum()
    tangent_loss = tangent_loss + (w_len.double().cpu() * tangents[2].double().cpu()).sum()

    def joint_loss(packed: torch.Tensor) -> torch.Tensor:
        return _epc_analytic_loss(raydn, packed[:1], packed[1:], weights)

    packed = torch.cat([source, receiver], dim=0)
    direction = torch.cat([v_source, v_receiver], dim=0)
    fd_value = central_difference_directional(
        joint_loss, packed, direction, FD_STEP_POSITION
    )
    assert relative_error(tangent_loss, fd_value, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_refl_epc_jvp_vjp_inner_product_duality():
    raydn, vertices = _wall_scene()
    source, receiver = _epc_endpoints()
    tape_prim_id, tape_barycentric, tape_t = _epc_tape(raydn, source, receiver)
    assert int(tape_prim_id.min()) >= 0

    generator = torch.Generator(device="cpu").manual_seed(41)
    u_real = torch.randn(1, generator=generator).to("cuda")
    u_imag = torch.randn(1, generator=generator).to("cuda")
    u_len = torch.randn(1, generator=generator).to("cuda")
    v_source = torch.randn(source.shape, generator=generator).to("cuda")
    v_receiver = torch.randn(receiver.shape, generator=generator).to("cuda")
    v_vertices = torch.randn(vertices.shape, generator=generator).to("cuda")

    tangents = ops.raydn_refl_epc_jvp(
        raydn,
        source,
        receiver,
        None,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        tangent_vertices=v_vertices,
        tangent_source=v_source,
        tangent_receiver=v_receiver,
    )
    lhs = (tangents[0].double() * u_real.double()).sum()
    lhs = lhs + (tangents[1].double() * u_imag.double()).sum()
    lhs = lhs + (tangents[2].double() * u_len.double()).sum()

    grads = ops.raydn_refl_epc_backward(
        raydn,
        source,
        receiver,
        None,
        tape_prim_id,
        tape_barycentric,
        tape_t,
        grad_field_real=u_real,
        grad_field_imag=u_imag,
        grad_path_length=u_len,
        need_grad_vertices=True,
        need_grad_source=True,
        need_grad_receiver=True,
    )
    rhs = (grads[0].double() * v_vertices.double()).sum()
    rhs = rhs + (grads[1].double() * v_source.double()).sum()
    rhs = rhs + (grads[2].double() * v_receiver.double()).sum()

    assert relative_error(lhs.cpu(), rhs.cpu(), abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_refl_epc_field_ad_reports_invalid_paths_with_zero_grads():
    """RayD's EPC discovery forward cannot validate single-plane fixtures
    (its winner tape stays empty); the autograd.Function must then yield
    exact zero gradients instead of failing or fabricating gradients."""

    raydn, vertices = _wall_scene()
    source, receiver = _epc_endpoints()

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    source_ad = source.clone().requires_grad_(True)
    receiver_ad = receiver.clone().requires_grad_(True)
    out = ops.raydn_refl_epc_field_ad(
        raydn, vertices_ad, source_ad, receiver_ad, None, 1
    )
    assert not bool(out["valid"].any())
    loss = out["field_real"].sum() + out["field_imag"].sum()
    loss = loss + out["path_length"].sum()
    loss.backward()
    assert source_ad.grad is not None and float(source_ad.grad.abs().max()) == 0.0
    assert receiver_ad.grad is not None and float(receiver_ad.grad.abs().max()) == 0.0
    assert vertices_ad.grad is not None and float(vertices_ad.grad.abs().max()) == 0.0


def test_fixed_winner_tape_outputs_are_non_differentiable():
    raydn, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    source, receiver = _epc_endpoints()

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    ray_o_ad = ray_o.clone().requires_grad_(True)
    source_ad = source.clone().requires_grad_(True)
    receiver_ad = receiver.clone().requires_grad_(True)

    intersect_out = ops.raydn_intersect_ad(
        raydn, vertices_ad, ray_o_ad, ray_d, _empty_tmax()
    )
    assert intersect_out["t"].requires_grad
    assert intersect_out["p"].requires_grad
    for name in ("shape_id", "prim_id", "local_prim_id", "global_prim_id"):
        assert not intersect_out[name].requires_grad
        assert intersect_out[name].grad_fn is None

    reflection_out = ops.raydn_trace_reflections_ad(
        raydn, vertices_ad, ray_o_ad, ray_d, _empty_tmax(), None, 1
    )
    assert reflection_out["t"].requires_grad
    assert reflection_out["image_sources"].requires_grad
    for name in (
        "valid",
        "prim_ids",
        "tape_barycentric",
        "tape_hit_points",
        "tape_normals",
    ):
        assert not reflection_out[name].requires_grad
        assert reflection_out[name].grad_fn is None

    epc_out = ops.raydn_refl_epc_field_ad(
        raydn, vertices_ad, source_ad, receiver_ad, None, 1
    )
    for name in ("field_real", "field_imag", "path_length"):
        assert epc_out[name].requires_grad
    for name in ("valid", "resolved_prim_id", "tape_prim_id", "tape_barycentric"):
        assert not epc_out[name].requires_grad
        assert epc_out[name].grad_fn is None
