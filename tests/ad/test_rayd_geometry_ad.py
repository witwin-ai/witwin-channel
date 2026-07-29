# Copyright Xingyu Chen.
# AD-A0: RayD fixed-winner geometry JVP/VJP versus central finite differences.

"""AD-A0: RayD fixed-winner geometry JVP/VJP versus central finite differences."""

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
from witwin.core import Mesh, PhysicalMaterial, Scene, Structure
from witwin.channel.kernels import geometry as ops
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for RayD geometry AD"
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
    from witwin.channel.deployment import build_info

    try:
        return build_info()["rayd_integration"] == "source-linked"
    except ModuleNotFoundError:
        return False


def _build_rayd_scene(vertices: torch.Tensor, faces) -> tuple[object, torch.Tensor]:
    """Build a single-structure RayD scene; returns (scene wrapper, global vertices)."""

    if not _source_linked_rayd_available():
        pytest.skip("RayD native extension is not built")
    scene = Scene(
        structures=(
            Structure(
                Mesh(
                    vertices.detach().cpu().to(torch.float32),
                    torch.tensor(faces, dtype=torch.int32),
                    recenter=False,
                    fill_mode="surface",
                    topology_diagnostics=False,
                ),
                PhysicalMaterial(eps_r=2.0),
            ),
        ),
        endpoints=(),
    )
    compiled = compile_scene(
        scene,
        reference_frequency_hz=3.5e9,
    )
    rayd = compiled.rayd
    return rayd, rayd.mesh_tensors[0][0]


def _triangle_scene() -> tuple[object, torch.Tensor]:
    return _build_rayd_scene(
        torch.tensor(_TILTED_TRIANGLE_VERTICES, dtype=torch.float32), _TRIANGLE_FACES
    )


def _wall_scene() -> tuple[object, torch.Tensor]:
    return _build_rayd_scene(
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
    rayd: object, ray_o: torch.Tensor, ray_d: torch.Tensor, w_t: torch.Tensor, w_p: torch.Tensor,
) -> torch.Tensor:
    hit = ops.rayd_intersect_forward(
        rayd, ray_o.contiguous(), ray_d, _empty_tmax(), None, flags=7
    )
    loss = (w_t.double() * hit["t"].double()).sum()
    loss = loss + (w_p.double() * hit["p"].double()).sum()
    return loss


def test_intersect_vjp_matches_fd_wrt_ray_origin():
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    w_t, w_p = _intersect_loss_weights(ray_o.shape[0])

    ray_o_ad = ray_o.clone().requires_grad_(True)
    out = ops.rayd_intersect_ad(rayd, vertices, ray_o_ad, ray_d, _empty_tmax())
    loss = (w_t * out["t"]).sum() + (w_p * out["p"]).sum()
    loss.backward()
    assert ray_o_ad.grad is not None

    fd_grad = central_difference_gradient(
        lambda x: _intersect_fd_loss(rayd, x, ray_d, w_t, w_p),
        ray_o,
        FD_STEP_POSITION,
    )
    assert (
        relative_error(ray_o_ad.grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


def test_intersect_vjp_matches_fd_wrt_vertices():
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    w_t, w_p = _intersect_loss_weights(ray_o.shape[0])

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    out = ops.rayd_intersect_ad(rayd, vertices_ad, ray_o, ray_d, _empty_tmax())
    loss = (w_t * out["t"]).sum() + (w_p * out["p"]).sum()
    loss.backward()
    assert vertices_ad.grad is not None

    def rebuild_loss(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_rayd_scene(perturbed_vertices, _TRIANGLE_FACES)
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
    rayd, vertices = _triangle_scene()
    del vertices
    ray_o, ray_d = _triangle_rays()
    hit = ops.rayd_intersect_forward(rayd, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(13)
    direction = torch.randn(ray_o.shape, generator=generator).to("cuda")
    tangents = ops.rayd_intersect_jvp(
        rayd,
        ray_o,
        ray_d,
        None,
        hit["global_prim_id"],
        hit["barycentric"],
        tangent_ray_o=direction,
    )

    def forward_t(x: torch.Tensor) -> torch.Tensor:
        return ops.rayd_intersect_forward(
            rayd, x.contiguous(), ray_d, _empty_tmax(), None, flags=7
        )["t"]

    def forward_p(x: torch.Tensor) -> torch.Tensor:
        return ops.rayd_intersect_forward(
            rayd, x.contiguous(), ray_d, _empty_tmax(), None, flags=7
        )["p"]

    fd_t = central_difference_directional(forward_t, ray_o, direction, FD_STEP_POSITION)
    fd_p = central_difference_directional(forward_p, ray_o, direction, FD_STEP_POSITION)
    assert relative_error(tangents[0], fd_t, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    assert relative_error(tangents[1], fd_p, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_intersect_jvp_matches_fd_wrt_vertices():
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.rayd_intersect_forward(rayd, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(17)
    direction = torch.randn(vertices.shape, generator=generator).to("cuda")
    tangents = ops.rayd_intersect_jvp(
        rayd,
        ray_o,
        ray_d,
        None,
        hit["global_prim_id"],
        hit["barycentric"],
        tangent_vertices=direction,
    )

    def rebuild_t(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_rayd_scene(perturbed_vertices, _TRIANGLE_FACES)
        return ops.rayd_intersect_forward(
            rebuilt, ray_o, ray_d, _empty_tmax(), None, flags=7
        )["t"]

    def rebuild_p(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_rayd_scene(perturbed_vertices, _TRIANGLE_FACES)
        return ops.rayd_intersect_forward(
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
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.rayd_intersect_forward(rayd, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(19)
    direction = torch.randn(ray_o.shape, generator=generator).to("cuda")
    expected = ops.rayd_intersect_jvp(
        rayd,
        ray_o,
        ray_d,
        None,
        hit["global_prim_id"],
        hit["barycentric"],
        tangent_ray_o=direction,
    )

    with torch.autograd.forward_ad.dual_level():
        dual_ray_o = torch.autograd.forward_ad.make_dual(ray_o, direction)
        out = ops.rayd_intersect_ad(rayd, vertices, dual_ray_o, ray_d, _empty_tmax())
        _, tangent_t = torch.autograd.forward_ad.unpack_dual(out["t"])
        _, tangent_p = torch.autograd.forward_ad.unpack_dual(out["p"])

    assert tangent_t is not None and tangent_p is not None
    assert relative_error(tangent_t, expected[0], abs_floor=ABS_TOL) <= REL_TOL_PATH
    assert relative_error(tangent_p, expected[1], abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_intersect_jvp_vjp_inner_product_duality():
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.rayd_intersect_forward(rayd, ray_o, ray_d, _empty_tmax(), None, flags=7)

    generator = torch.Generator(device="cpu").manual_seed(23)
    u_t = torch.randn(ray_o.shape[0], generator=generator).to("cuda")
    u_p = torch.randn(ray_o.shape[0], 3, generator=generator).to("cuda")
    v_ray_o = torch.randn(ray_o.shape, generator=generator).to("cuda")
    v_vertices = torch.randn(vertices.shape, generator=generator).to("cuda")

    # <J v, u> from the native JVP.
    tangents = ops.rayd_intersect_jvp(
        rayd,
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
    out = ops.rayd_intersect_ad(rayd, vertices_ad, ray_o_ad, ray_d, _empty_tmax())
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
    rayd: object, ray_o: torch.Tensor, ray_d: torch.Tensor, w_t: torch.Tensor, w_img: torch.Tensor,
) -> torch.Tensor:
    forward = ops.rayd_trace_reflections_forward_tape(
        rayd, ray_o.contiguous(), ray_d, _empty_tmax(), None, 1
    )
    loss = (w_t.double() * forward[1].double()).sum()
    loss = loss + (w_img.double() * forward[2].double()).sum()
    return loss


def test_reflection_chain_vjp_matches_fd_wrt_ray_origin():
    rayd, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    w_t, w_img = _reflection_loss_weights()

    ray_o_ad = ray_o.clone().requires_grad_(True)
    out = ops.rayd_trace_reflections_ad(
        rayd, vertices, ray_o_ad, ray_d, _empty_tmax(), None, 1
    )
    assert bool(out["valid"].all())
    loss = (w_t * out["t"]).sum() + (w_img * out["image_sources"]).sum()
    loss.backward()
    assert ray_o_ad.grad is not None

    fd_grad = central_difference_gradient(
        lambda x: _reflection_fd_loss(rayd, x, ray_d, w_t, w_img),
        ray_o,
        FD_STEP_POSITION,
    )
    assert (
        relative_error(ray_o_ad.grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


def test_reflection_chain_vjp_matches_fd_wrt_vertices():
    rayd, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    w_t, w_img = _reflection_loss_weights()

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    out = ops.rayd_trace_reflections_ad(
        rayd, vertices_ad, ray_o, ray_d, _empty_tmax(), None, 1
    )
    assert bool(out["valid"].all())
    loss = (w_t * out["t"]).sum() + (w_img * out["image_sources"]).sum()
    loss.backward()
    assert vertices_ad.grad is not None

    def rebuild_loss(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_rayd_scene(perturbed_vertices, _WALL_FACES)
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
    rayd, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    forward = ops.rayd_trace_reflections_forward_tape(
        rayd, ray_o, ray_d, _empty_tmax(), None, 1
    )
    _valid, t, image_sources, _prim, tape_prim_id, tape_bary, tape_hits, tape_normals, _active = forward

    generator = torch.Generator(device="cpu").manual_seed(47)
    u_t = torch.randn(t.shape, generator=generator).to("cuda")
    u_img = torch.randn(image_sources.shape, generator=generator).to("cuda")
    v_ray_o = torch.randn(ray_o.shape, generator=generator).to("cuda")
    v_vertices = torch.randn(vertices.shape, generator=generator).to("cuda")

    tangents = ops.rayd_trace_reflections_jvp(
        rayd,
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

    grads = ops.rayd_trace_reflections_backward(
        rayd,
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


def test_fixed_winner_tape_outputs_are_non_differentiable():
    rayd, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    ray_o_ad = ray_o.clone().requires_grad_(True)

    intersect_out = ops.rayd_intersect_ad(
        rayd, vertices_ad, ray_o_ad, ray_d, _empty_tmax()
    )
    assert intersect_out["t"].requires_grad
    assert intersect_out["p"].requires_grad
    for name in ("shape_id", "prim_id", "local_prim_id", "global_prim_id"):
        assert not intersect_out[name].requires_grad
        assert intersect_out[name].grad_fn is None

    reflection_out = ops.rayd_trace_reflections_ad(
        rayd, vertices_ad, ray_o_ad, ray_d, _empty_tmax(), None, 1
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


def test_intersect_vjp_matches_fd_wrt_normal_and_barycentric_cotangents():
    """Nonzero cotangents on the n and barycentric outputs (the gn/gbary
 adjoint kernel paths) versus scene-rebuild central FD."""

    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    generator = torch.Generator(device="cpu").manual_seed(53)
    w_n = torch.randn(ray_o.shape[0], 3, generator=generator).to("cuda")
    w_b = torch.randn(ray_o.shape[0], 3, generator=generator).to("cuda")

    vertices_ad = vertices.detach().clone().requires_grad_(True)
    out = ops.rayd_intersect_ad(rayd, vertices_ad, ray_o, ray_d, _empty_tmax())
    loss = (w_n * out["n"]).sum() + (w_b * out["barycentric"]).sum()
    loss.backward()
    assert vertices_ad.grad is not None
    assert float(vertices_ad.grad.abs().max()) > 0.0

    def rebuild_loss(perturbed_vertices: torch.Tensor) -> torch.Tensor:
        rebuilt, _ = _build_rayd_scene(perturbed_vertices, _TRIANGLE_FACES)
        hit = ops.rayd_intersect_forward(
            rebuilt, ray_o, ray_d, _empty_tmax(), None, flags=7
        )
        loss = (w_n.double() * hit["n"].double()).sum()
        return loss + (w_b.double() * hit["barycentric"].double()).sum()

    generator = torch.Generator(device="cpu").manual_seed(59)
    for _ in range(3):
        direction = torch.randn(vertices.shape, generator=generator).to("cuda")
        fd_value = central_difference_directional(
            rebuild_loss, vertices, direction, FD_STEP_GEOMETRY
        )
        ad_value = (vertices_ad.grad.double() * direction.double()).sum().cpu()
        assert relative_error(ad_value, fd_value, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_intersect_vjp_matches_fd_wrt_ray_direction():
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    w_t, w_p = _intersect_loss_weights(ray_o.shape[0])

    ray_d_ad = ray_d.clone().requires_grad_(True)
    out = ops.rayd_intersect_ad(rayd, vertices, ray_o, ray_d_ad, _empty_tmax())
    loss = (w_t * out["t"]).sum() + (w_p * out["p"]).sum()
    loss.backward()
    assert ray_d_ad.grad is not None
    assert float(ray_d_ad.grad.abs().max()) > 0.0

    fd_grad = central_difference_gradient(
        lambda d: _intersect_fd_loss(rayd, ray_o, d.contiguous(), w_t, w_p),
        ray_d,
        FD_STEP_POSITION,
    )
    assert (
        relative_error(ray_d_ad.grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


def test_reflection_chain_functional_jvp_matches_native():
    """Exercise _RaydTraceReflectionsAdFunction.jvp through torch.func.jvp."""

    rayd, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    forward = ops.rayd_trace_reflections_forward_tape(
        rayd, ray_o, ray_d, _empty_tmax(), None, 1
    )
    _valid, _t, image_sources, _prim, tape_prim_id, tape_bary, tape_hits, tape_normals, _active = forward

    generator = torch.Generator(device="cpu").manual_seed(61)
    v_ray_o = torch.randn(ray_o.shape, generator=generator).to("cuda")
    expected = ops.rayd_trace_reflections_jvp(
        rayd,
        ray_o,
        ray_d,
        None,
        tape_prim_id,
        tape_bary,
        tape_hits,
        tape_normals,
        image_sources,
        tangent_ray_o=v_ray_o,
    )

    def f(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = ops.rayd_trace_reflections_ad(
            rayd, vertices, x, ray_d, _empty_tmax(), None, 1
        )
        return out["t"], out["image_sources"]

    _primals, tangents = torch.func.jvp(f, (ray_o,), (v_ray_o,))
    assert float(expected[0].abs().max()) > 0.0
    assert relative_error(tangents[0], expected[0], abs_floor=ABS_TOL) <= REL_TOL_PATH
    assert relative_error(tangents[1], expected[1], abs_floor=ABS_TOL) <= REL_TOL_PATH


def test_jvp_facades_reject_wrong_shaped_tangents():
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.rayd_intersect_forward(rayd, ray_o, ray_d, _empty_tmax(), None, flags=7)

    bad_width = torch.zeros(
        (ray_o.shape[0], 2), dtype=torch.float32, device="cuda"
    )
    with pytest.raises(ValueError):
        ops.rayd_intersect_jvp(
            rayd,
            ray_o,
            ray_d,
            None,
            hit["global_prim_id"],
            hit["barycentric"],
            tangent_ray_o=bad_width,
        )

    bad_rows = torch.zeros(
        (ray_o.shape[0] + 1, 3), dtype=torch.float32, device="cuda"
    )
    with pytest.raises(ValueError):
        ops.rayd_intersect_jvp(
            rayd,
            ray_o,
            ray_d,
            None,
            hit["global_prim_id"],
            hit["barycentric"],
            tangent_ray_d=bad_rows,
        )

    # A vertex tangent that is not the scene global vertex table is rejected
    # by the native entry before any kernel launch.
    bad_vertices = torch.zeros(
        (vertices.shape[0] + 5, 3), dtype=torch.float32, device="cuda"
    )
    with pytest.raises(RuntimeError):
        ops.rayd_intersect_jvp(
            rayd,
            ray_o,
            ray_d,
            None,
            hit["global_prim_id"],
            hit["barycentric"],
            tangent_vertices=bad_vertices,
        )


def test_backward_and_jvp_facades_reject_mismatched_tape_batch():
    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    hit = ops.rayd_intersect_forward(rayd, ray_o, ray_d, _empty_tmax(), None, flags=7)
    grad_t = torch.ones(ray_o.shape[0], dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError):
        ops.rayd_intersect_backward(
            rayd,
            ray_o,
            ray_d,
            _empty_tmax(),
            None,
            hit["global_prim_id"][:1],
            hit["barycentric"][:1],
            grad_t=grad_t,
            need_grad_ray_o=True,
        )
    with pytest.raises(ValueError):
        ops.rayd_intersect_jvp(
            rayd,
            ray_o,
            ray_d,
            None,
            hit["global_prim_id"][:1],
            hit["barycentric"][:1],
            tangent_ray_o=torch.zeros_like(ray_o),
        )


def test_reflection_facades_reject_mismatched_tape_batch():
    rayd, vertices = _wall_scene()
    ray_o, ray_d = _wall_reflection_rays()
    forward = ops.rayd_trace_reflections_forward_tape(
        rayd, ray_o, ray_d, _empty_tmax(), None, 1
    )
    _valid, _t, image_sources, _prim, tape_prim_id, tape_bary, tape_hits, tape_normals, _active = forward

    # Batch of 2 rays against the length-1 tape from the single-ray forward.
    ray_o2 = ray_o.repeat(2, 1).contiguous()
    ray_d2 = ray_d.repeat(2, 1).contiguous()
    grad_t2 = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        ops.rayd_trace_reflections_backward(
            rayd,
            ray_o2,
            ray_d2,
            _empty_tmax(),
            None,
            tape_prim_id,
            tape_bary,
            tape_hits,
            tape_normals,
            image_sources,
            grad_t=grad_t2,
        )
    with pytest.raises(ValueError):
        ops.rayd_trace_reflections_jvp(
            rayd,
            ray_o2,
            ray_d2,
            None,
            tape_prim_id,
            tape_bary,
            tape_hits,
            tape_normals,
            image_sources,
            tangent_ray_o=torch.zeros_like(ray_o2),
        )


def test_composed_functorch_transforms_raise_not_implemented():
    """torch.func.grad over forward-mode jvp (the HVP recipe) must fail
 loudly instead of silently returning zeros (the AD contract)."""

    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    generator = torch.Generator(device="cpu").manual_seed(71)
    direction = torch.randn(ray_o.shape, generator=generator).to("cuda")

    def scalar_loss(x: torch.Tensor) -> torch.Tensor:
        out = ops.rayd_intersect_ad(rayd, vertices, x, ray_d, _empty_tmax())
        return out["t"].sum()

    def jvp_scalar(x: torch.Tensor) -> torch.Tensor:
        _, tangent = torch.func.jvp(scalar_loss, (x,), (direction,))
        return tangent

    with pytest.raises(NotImplementedError):
        torch.func.grad(jvp_scalar)(ray_o)

    def dual_scalar(x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(x, direction)
            out = ops.rayd_intersect_ad(rayd, vertices, dual, ray_d, _empty_tmax())
            return torch.autograd.forward_ad.unpack_dual(out["t"]).tangent.sum()

    with pytest.raises(NotImplementedError):
        torch.func.grad(dual_scalar)(ray_o)


def test_double_backward_raises():
    """create_graph=True through the once-differentiable backwards must raise
 instead of silently dropping second-order contributions.

 first-order differentiation moves the raise to the request itself: it now fires inside the
 backward that ``create_graph=True`` asked to be differentiable, before any
 native companion launches, and names the owner rather than Torch.
 """

    rayd, vertices = _triangle_scene()
    ray_o, ray_d = _triangle_rays()
    ray_o_ad = ray_o.clone().requires_grad_(True)
    out = ops.rayd_intersect_ad(rayd, vertices, ray_o_ad, ray_d, _empty_tmax())
    with pytest.raises(NotImplementedError, match="first-order only") as raised:
        torch.autograd.grad(out["t"].sum(), ray_o_ad, create_graph=True)
    assert "_RaydIntersectAdFunction.backward" in str(raised.value)

    wall, wall_vertices = _wall_scene()
    refl_ray_o, refl_ray_d = _wall_reflection_rays()
    refl_ray_o_ad = refl_ray_o.clone().requires_grad_(True)
    refl_out = ops.rayd_trace_reflections_ad(
        wall, wall_vertices, refl_ray_o_ad, refl_ray_d, _empty_tmax(), None, 1
    )
    with pytest.raises(NotImplementedError, match="first-order only"):
        torch.autograd.grad(
            refl_out["t"].sum(), refl_ray_o_ad, create_graph=True
        )