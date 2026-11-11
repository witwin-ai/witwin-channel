# Copyright Xingyu Chen.
# Tests fixed-topology BDPT AD through the fused wavefront: primal, FD, duality and export.

"""Tests fixed-topology BDPT AD through the fused wavefront."""

from __future__ import annotations

import pytest
import torch

import witwin.channel.montecarlo.bdpt as bdpt
from tests.ad._fd import relative_error
from tests.ad._tolerances import ABS_TOL, REL_TOL_GENERAL
from tests.support.core_world import make_mesh_structure, make_receiver, make_transmitter
from tests.support.scenes import rough_wall_structure
from witwin.channel.deployment import build_info
from witwin.channel.scene import compile as compile_scene
from witwin.core import MaterialLayer, PhysicalMaterial, Scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

_FREQ = 3.0e9
_SAMPLES = 16384


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native transmission is not built")


def _wall(material, x: float, surface_id: int):
    return make_mesh_structure(
        vertices=torch.tensor([[x, -4.0, -4.0], [x, 4.0, -4.0], [x, -4.0, 4.0], [x, 4.0, 4.0]]),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name=f"wall-{surface_id}",
        surface_id=surface_id,
    )


def _mixed_scene(frequency=_FREQ) -> Scene:
    """Two-layer lossy front wall and a lossy back wall: transmit/reflect mixed chains."""

    front = PhysicalMaterial(
        layers=(
            MaterialLayer(thickness_m=0.1, eps_r=4.0, sigma_e=0.05),
            MaterialLayer(thickness_m=0.05, eps_r=2.5, sigma_e=0.01),
        ),
        name="front",
    )
    back = PhysicalMaterial(eps_r=6.0, sigma_e=0.3, name="back")
    return Scene(
        structures=[_wall(front, 2.5, 1), _wall(back, 6.0, 2)],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 0.0])),
            make_transmitter(position=torch.tensor([0.0, 0.3, 0.0])),
            *(
                make_receiver(position=torch.tensor([5.0, y, z]))
                for y, z in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
            ),
        ],
        metadata={"frequency": frequency},
    )


def _rough_scene(frequency=_FREQ) -> Scene:
    """A rough wall in front of a lossy back wall: scatter vertices and continued subpaths."""

    rough = rough_wall_structure(
        2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=1.0, eps_r=4.0, sigma_e=0.05
    )
    back = PhysicalMaterial(eps_r=6.0, sigma_e=0.3, name="back")
    return Scene(
        structures=[rough, _wall(back, -1.5, 9)],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.0])),
            make_receiver(position=torch.tensor([0.5, 1.0, 0.3])),
        ],
        metadata={"frequency": frequency},
    )


_MIXED = dict(components={"transmission"}, max_depth=3)
_ROUGH = dict(components={"scattering", "transmission"}, max_depth=4, max_scattering_order=3)


def _solve(scene, ad_mode, kw, *, seed=5, **extra):
    config = bdpt.Config(samples=_SAMPLES, seed=seed, ad_mode=ad_mode, **kw, **extra)
    return bdpt.solve(scene, config, reference_frequency_hz=scene.metadata["frequency"])


def _materials(compiled):
    return compiled.materials


def _table(compiled):
    return next(iter(compiled.rough_material_runtimes.values())).table


def _direction(base: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device=base.device).manual_seed(3)
    return torch.rand(base.shape, generator=generator, device=base.device, dtype=base.dtype) + 0.5


def _vjp_directional(scene, kw, holder, name, direction, *, seed=5, **extra):
    base = getattr(holder, name)
    base.requires_grad_(True)
    try:
        result = _solve(scene, "vjp", kw, seed=seed, **extra)
        result.path_gain.sum().backward()
        grad = base.grad.detach().clone()
    finally:
        base.requires_grad_(False)
        base.grad = None
    return (grad.double() * direction.double()).sum(), result


def _jvp_directional(scene, kw, holder, name, direction):
    base = getattr(holder, name)
    with torch.autograd.forward_ad.dual_level():
        object.__setattr__(
            holder, name, torch.autograd.forward_ad.make_dual(base.detach().clone(), direction)
        )
        try:
            result = _solve(scene, "jvp", kw)
        finally:
            object.__setattr__(holder, name, base)
        tangent = torch.autograd.forward_ad.unpack_dual(result.path_gain).tangent
    assert tangent is not None
    return tangent.double().sum(), result


def _fd_directional(scene, kw, holder, name, direction, step, *, seeds=(5,)):
    base = getattr(holder, name)
    original = base.detach().clone()
    values = []
    try:
        for sign in (1.0, -1.0):
            with torch.no_grad():
                base.copy_(original + sign * step * direction)
            total = torch.zeros((), dtype=torch.float64, device="cuda")
            for seed in seeds:
                total = total + _solve(scene, "none", kw, seed=seed).path_gain.double().sum()
            values.append(total / len(seeds))
    finally:
        with torch.no_grad():
            base.copy_(original)
    return (values[0] - values[1]) / (2.0 * step)


@pytest.mark.parametrize("builder,kw", ((_mixed_scene, _MIXED), (_rough_scene, _ROUGH)))
@pytest.mark.parametrize("ad_mode", ("jvp", "vjp"))
def test_wavefront_ad_primal_equals_none_bitwise(builder, kw, ad_mode):
    _require_native()
    scene = builder()
    compiled = compile_scene(scene, reference_frequency_hz=_FREQ)
    none = _solve(scene, "none", kw)
    compiled.materials.eps_r.requires_grad_(ad_mode == "vjp")
    try:
        ad = _solve(scene, ad_mode, kw)
    finally:
        compiled.materials.eps_r.requires_grad_(False)
    assert ad.metadata["sampler"] == "wavefront"
    assert float(none.path_gain.sum()) > 0.0
    torch.testing.assert_close(ad.path_gain.detach(), none.path_gain, rtol=0.0, atol=0.0)


# Leaves that do not move the frozen event probabilities give a same-seed FD.
@pytest.mark.parametrize(
    "builder,kw,holder,name,step",
    (
        (_mixed_scene, _MIXED, _materials, "eps_r", 1.0e-2),
        (_mixed_scene, _MIXED, _materials, "sigma_e", 1.0e-3),
        (_rough_scene, _ROUGH, _table, "f_te", 1.0e-3),
        (_rough_scene, _ROUGH, _table, "f_tm", 1.0e-3),
        (_rough_scene, _ROUGH, _materials, "eps_r", 1.0e-2),
    ),
)
def test_wavefront_vjp_matches_same_seed_fd(builder, kw, holder, name, step):
    _require_native()
    scene = builder()
    leaf_holder = holder(compile_scene(scene, reference_frequency_hz=_FREQ))
    direction = _direction(getattr(leaf_holder, name))
    ad, _ = _vjp_directional(scene, kw, leaf_holder, name, direction)
    fd = _fd_directional(scene, kw, leaf_holder, name, direction, step)
    assert abs(float(ad)) > 0.0
    assert relative_error(ad, fd, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_wavefront_layer_vjp_matches_fd_of_the_seed_mean():
    # Layer leaves also move the frozen event probabilities, so the same-seed
    # estimate is not differentiable sample by sample; its seed mean is.
    _require_native()
    scene = _mixed_scene()
    materials = compile_scene(scene, reference_frequency_hz=_FREQ).materials
    direction = _direction(materials.layer_eps_r)
    seeds = tuple(range(8))
    ad = sum(
        _vjp_directional(scene, _MIXED, materials, "layer_eps_r", direction, seed=seed)[0]
        for seed in seeds
    ) / len(seeds)
    fd = _fd_directional(scene, _MIXED, materials, "layer_eps_r", direction, 1.0e-2, seeds=seeds)
    assert relative_error(ad, fd, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize(
    "builder,kw,holder,name",
    (
        (_mixed_scene, _MIXED, _materials, "eps_r"),
        (_mixed_scene, _MIXED, _materials, "sigma_e"),
        (_mixed_scene, _MIXED, _materials, "layer_eps_r"),
        (_mixed_scene, _MIXED, _materials, "layer_sigma_e"),
        (_mixed_scene, _MIXED, _materials, "layer_thickness_m"),
        (_rough_scene, _ROUGH, _table, "f_te"),
        (_rough_scene, _ROUGH, _table, "f_tm"),
        (_rough_scene, _ROUGH, _materials, "eps_r"),
        (_rough_scene, _ROUGH, _materials, "sigma_e"),
    ),
)
def test_wavefront_jvp_and_vjp_are_dual(builder, kw, holder, name):
    _require_native()
    scene = builder()
    leaf_holder = holder(compile_scene(scene, reference_frequency_hz=_FREQ))
    direction = _direction(getattr(leaf_holder, name))
    forward, _ = _jvp_directional(scene, kw, leaf_holder, name, direction)
    reverse, _ = _vjp_directional(scene, kw, leaf_holder, name, direction)
    assert abs(float(reverse)) > 0.0
    assert relative_error(forward, reverse, abs_floor=ABS_TOL) <= 1.0e-4


@pytest.mark.parametrize("builder,kw", ((_mixed_scene, _MIXED), (_rough_scene, _ROUGH)))
def test_wavefront_frequency_jvp_and_vjp_are_dual(builder, kw):
    _require_native()
    tangent = torch.tensor(1.0e6, device="cuda", dtype=torch.float64)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            torch.tensor(_FREQ, device="cuda", dtype=torch.float64), tangent
        )
        result = _solve(builder(dual), "jvp", kw)
        forward = torch.autograd.forward_ad.unpack_dual(result.path_gain).tangent.double().sum()
    frequency = torch.tensor(_FREQ, device="cuda", dtype=torch.float64, requires_grad=True)
    _solve(builder(frequency), "vjp", kw).path_gain.sum().backward()
    reverse = frequency.grad * tangent
    assert abs(float(reverse)) > 0.0
    assert relative_error(forward, reverse, abs_floor=ABS_TOL) <= 1.0e-4


def test_wavefront_vjp_with_exported_paths_keeps_primal_and_gradient():
    # With export_paths the estimate comes from the accumulated rows; the
    # wavefront companion adds the same gradient without moving the primal.
    _require_native()
    scene = _mixed_scene()
    materials = compile_scene(scene, reference_frequency_hz=_FREQ).materials
    direction = _direction(materials.eps_r)
    exported_none = _solve(scene, "none", _MIXED, export_paths=True)
    plain, _ = _vjp_directional(scene, _MIXED, materials, "eps_r", direction)
    exported, result = _vjp_directional(
        scene, _MIXED, materials, "eps_r", direction, export_paths=True
    )
    torch.testing.assert_close(
        result.path_gain.detach(), exported_none.path_gain, rtol=0.0, atol=0.0
    )
    assert relative_error(exported, plain, abs_floor=ABS_TOL) <= 1.0e-5
