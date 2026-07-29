# Copyright Xingyu Chen.
# Tests bdpt solver ad.

"""ADR-022 solver-level acceptance gates for BDPT full fixed-topology AD.

Drives the public ``montecarlo.bdpt`` solver with ``ad_mode`` in
``{"none","jvp","vjp"}`` and pins the ADR-022 acceptance protocol:

1. ``ad_mode="none"`` bitwise (seed-deterministic self-comparison run-to-run) with
   ``ad_status == "none"`` metadata, on a wedge+rough fixture.
2. Primal-under-ad bitwise equals the ``"none"`` primal (the AD wrappers call the
   same forward symbols).
3. Central-difference cross-checks at FIXED seed for the differentiable parameter
   set -- layer ``eps_r``/``sigma_e``/``thickness``, resident table values,
   carrier ``frequency``, ``tx_power`` -- on (a) an enumerated-dominated
   reflection fixture, (b) a mixed-chain shooting (transmission) fixture, (c) a
   scattering NEE fixture, (d) a coherent-combine fixture.
4. Unbiased-gradient sanity: the mean of per-seed AD gradients converges to the FD
   gradient of the seed-mean (validates the frozen-pdf estimator commutation).
5. Geometry-gradient refusal through the stochastic sampler
   (``ad_geometry == "enumerated_blocks_only"``; requesting a mesh/endpoint
   gradient that only reaches the shooting walk fails loudly).
6. JVP-vs-VJP consistency at the solver level.

Every FD uses the shared ``tests/ad/_fd`` engine with FIXED seed, so the frozen
sampling / masks / seeds make the estimator differentiable in the material / EM /
frequency / power leaves (ADR-022 fixed-topology contract). No tolerance is
weakened. These run after the supervisor rebuilds the extension with the
ADR-022 companions and threads ``ad_mode`` through the BDPT config.

Interface assumptions (documented; resolved by the ADR-022 section 7 wiring):

* ``tx_power`` becomes a live graph leaf under ``ad != "none"`` (today
  ``endpoints.transmitter_tensors`` reads ``float(power_w)``); the fixture puts a
  ``requires_grad`` tensor in ``Transmitter.power_w`` and expects the solver to
  thread it.
* ``result.metadata["ad_geometry"] == "enumerated_blocks_only"`` and geometry
  gradients that only reach the shooting sampler are refused loudly.
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
    FD_REL_STEP_FREQUENCY,
    FD_STEP_EPS_R,
    FD_STEP_SIGMA_E,
    FD_STEP_THICKNESS,
    REL_TOL_GENERAL,
)
from tests.support.scenes import (
    rough_wall_structure,
    same_side_wall_reflection_scene,
    transmission_wall_structure,
    wedge_diffraction_scene,
)
from witwin.core import Scene
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_transmitter,
)
from witwin.channel.deployment import build_info
from witwin.core import MaterialLayer, PhysicalMaterial
from witwin.channel.montecarlo.bdpt import Config as BDPTConfig
from witwin.channel.montecarlo.bdpt import solve as bdpt_solve
from witwin.channel.interactions.scattering import rough_material_runtimes
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for BDPT solver AD"
)

_FREQ = 3.0e9
_SEED = 7
_SAMPLES = 8192
_FREQUENCY_METADATA_KEY = "test_reference_frequency_hz"


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native diffraction/scattering is not built")


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _reflection_scene(frequency: float | torch.Tensor = _FREQ) -> Scene:
    # same_side (not single_wall): single_wall places the receiver at the mirror
    # image of the transmitter across the wall, a degenerate specular geometry
    # that yields zero reflection path gain (nothing to differentiate).
    base = same_side_wall_reflection_scene()
    return Scene(
        structures=base.structures,
        endpoints=base.endpoints,
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _transmission_scene(frequency: float | torch.Tensor = _FREQ) -> Scene:
    material = PhysicalMaterial(
        layers=(
            MaterialLayer(thickness_m=0.06, eps_r=4.0, sigma_e=0.02),
            MaterialLayer(thickness_m=0.09, eps_r=2.5, sigma_e=0.01),
        ),
        name="bdpt-thin-sheet",
    )
    return Scene(
        structures=[transmission_wall_structure(3.0, material)],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, 0.0, 0.0])),
            make_receiver(position=torch.tensor([6.0, 0.4, 0.2])),
        ],
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _scattering_scene(frequency: float | torch.Tensor = _FREQ) -> Scene:
    # rms_height/corr_length must stay inside the ensemble Kirchhoff validity
    # domain (k0*corr_length >= 6, RMS slope <= 0.5); corr_length_m=0.01 at 3 GHz
    # gives k0*l=0.63 and the table build refuses. Mirror the known-good ensemble
    # scattering geometry.
    wall = rough_wall_structure(
        2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=1.0,
        eps_r=4.0, sigma_e=0.05,
    )
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.0])),
        ],
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _coherent_scene(frequency: float | torch.Tensor = _FREQ) -> Scene:
    base = wedge_diffraction_scene(
        material=PhysicalMaterial(eps_r=5.0, sigma_e=0.02)
    )
    return Scene(
        structures=base.structures,
        endpoints=base.endpoints,
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _los_scene(frequency: float | torch.Tensor = _FREQ) -> Scene:
    # A wall that does not occlude the tx->rx line, so a los-only solve keeps
    # structures and routes the direct connection through the LoS companion
    # (_native_los_connection_samples), not the endpoint-only fast path.
    wall = same_side_wall_reflection_scene().structures[0]
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.5])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.5])),
        ],
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


_COMPONENTS = {
    "los": frozenset({"los"}),
    "reflection": frozenset({"reflection"}),
    "transmission": frozenset({"transmission"}),
    "scattering": frozenset({"scattering"}),
    "coherent": frozenset({"los", "reflection", "diffraction"}),
}


def _config(components, ad_mode, *, coherent=False, max_depth=1, seed=_SEED):
    return BDPTConfig(
        samples=_SAMPLES,
        seed=seed,
        max_depth=max_depth,
        components=components,
        coherent=coherent,
        ad_mode=ad_mode,
    )


def _reference_frequency(scene: Scene):
    return scene.metadata[_FREQUENCY_METADATA_KEY]


def _compile(scene: Scene):
    return compile_scene(
        scene,
        reference_frequency_hz=_reference_frequency(scene),
    )


def _solve(scene, components, ad_mode, *, coherent=False, max_depth=1, seed=_SEED):
    return bdpt_solve(
        scene,
        _config(
            components,
            ad_mode,
            coherent=coherent,
            max_depth=max_depth,
            seed=seed,
        ),
        reference_frequency_hz=_reference_frequency(scene),
    )


def _loss(result) -> torch.Tensor:
    return result.path_gain.sum()


# ---------------------------------------------------------------------------
# Gate 1 + 2: primal bitwise contracts.
# ---------------------------------------------------------------------------


def test_ad_mode_none_is_seed_deterministic_and_reports_none():
    _require_native()
    scene = _coherent_scene()
    a = _solve(scene, _COMPONENTS["coherent"], "none", coherent=True)
    b = _solve(scene, _COMPONENTS["coherent"], "none", coherent=True)
    torch.testing.assert_close(a.path_gain, b.path_gain, rtol=0.0, atol=0.0)
    assert a.metadata["kernel"]["ad_status"] == "none"
    assert a.path_gain.grad_fn is None and not a.path_gain.requires_grad


def test_primal_under_ad_equals_none_bitwise():
    _require_native()
    scene = _reflection_scene()
    compiled = _compile(scene)
    compiled.materials.eps_r.requires_grad_(True)
    try:
        none = _solve(scene, _COMPONENTS["reflection"], "none")
        vjp = _solve(scene, _COMPONENTS["reflection"], "vjp")
    finally:
        compiled.materials.eps_r.requires_grad_(False)
    assert not none.path_gain.requires_grad
    assert vjp.path_gain.requires_grad
    assert vjp.metadata["kernel"]["ad_status"] == "vjp"
    # AD wrappers dispatch the same forward symbols: primal values are bit-equal.
    torch.testing.assert_close(
        none.path_gain, vjp.path_gain.detach(), rtol=0.0, atol=0.0
    )


# ---------------------------------------------------------------------------
# Gate 3: FD cross-checks per fixture / differentiable parameter.
# ---------------------------------------------------------------------------


def _fd_via_store(scene, components, leaf, step, *, coherent=False):
    base = leaf.detach().clone()

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            leaf.copy_(values.to(device=leaf.device, dtype=leaf.dtype))
        try:
            result = _solve(scene, components, "none", coherent=coherent)
            return _loss(result).detach()
        finally:
            with torch.no_grad():
                leaf.copy_(base)

    return central_difference_gradient(evaluate, base, step)


def _material_leaf(scene, name):
    return getattr(_compile(scene).materials, name)


def _assert_leaf_grad_matches_fd(scene, components, leaf, step, *, coherent=False):
    leaf.requires_grad_(True)
    try:
        result = _solve(scene, components, "vjp", coherent=coherent)
        assert result.path_gain.requires_grad
        _loss(result).backward()
        assert leaf.grad is not None
        grad = leaf.grad.detach().clone()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert float(grad.abs().max()) > 0.0
    fd = _fd_via_store(scene, components, leaf, step, coherent=coherent)
    assert relative_error(grad, fd, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


@pytest.mark.parametrize("param,step", (("eps_r", FD_STEP_EPS_R), ("sigma_e", FD_STEP_SIGMA_E)))
def test_enumerated_reflection_material_grad_matches_fd(param, step):
    _require_native()
    scene = _reflection_scene()
    _assert_leaf_grad_matches_fd(
        scene, _COMPONENTS["reflection"], _material_leaf(scene, param), step
    )


@pytest.mark.parametrize(
    "param,step",
    (
        ("layer_eps_r", FD_STEP_EPS_R),
        ("layer_sigma_e", FD_STEP_SIGMA_E),
        ("layer_thickness_m", FD_STEP_THICKNESS),
    ),
)
def test_shooting_transmission_layer_grad_matches_fd(param, step):
    _require_native()
    scene = _transmission_scene()
    _assert_leaf_grad_matches_fd(
        scene, _COMPONENTS["transmission"], _material_leaf(scene, param), step
    )


def test_scattering_nee_table_grad_matches_fd():
    _require_native()
    scene = _scattering_scene()
    runtimes = rough_material_runtimes(_compile(scene))
    assert runtimes, "the scattering fixture must have a rough material"
    f_te = next(iter(runtimes.values())).table.f_te

    base = f_te.detach().clone()
    # The Kirchhoff table has 65536 entries, so a per-coordinate FD is not
    # tractable. Validate the VJP with a random-direction derivative instead:
    # <grad, d> must equal the central-difference directional derivative
    # (same rigour, one FD pair; no tolerance change).
    g = torch.Generator(device=f_te.device).manual_seed(0)
    direction = torch.randn(
        f_te.shape, generator=g, device=f_te.device, dtype=f_te.dtype
    )

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f_te.copy_(values.to(device=f_te.device, dtype=f_te.dtype))
        try:
            return _loss(_solve(scene, _COMPONENTS["scattering"], "none")).detach()
        finally:
            with torch.no_grad():
                f_te.copy_(base)

    # Step 1e-3 keeps the O(h^2) truncation of this norm-~256 direction well
    # below the tolerance (the AD gradient is exact as h -> 0).
    fd_directional = central_difference_directional(evaluate, base, direction, 1.0e-3)

    f_te.requires_grad_(True)
    try:
        result = _solve(scene, _COMPONENTS["scattering"], "vjp")
        assert result.path_gain.requires_grad
        _loss(result).backward()
        grad = f_te.grad.detach().clone()
    finally:
        f_te.requires_grad_(False)
        f_te.grad = None
    ad_directional = (grad.double() * direction.double()).sum()
    assert relative_error(
        ad_directional, fd_directional, abs_floor=ABS_TOL
    ) <= REL_TOL_GENERAL


def test_coherent_combine_material_grad_matches_fd():
    _require_native()
    scene = _coherent_scene()
    _assert_leaf_grad_matches_fd(
        scene, _COMPONENTS["coherent"], _material_leaf(scene, "eps_r"),
        FD_STEP_EPS_R, coherent=True,
    )


@pytest.mark.parametrize(
    "builder,components,coherent",
    (
        (_los_scene, _COMPONENTS["los"], False),
        (_reflection_scene, _COMPONENTS["reflection"], False),
        (_transmission_scene, _COMPONENTS["transmission"], False),
        (_scattering_scene, _COMPONENTS["scattering"], False),
    ),
)
def test_frequency_grad_matches_fd(builder, components, coherent):
    _require_native()
    frequency = torch.tensor(_FREQ, device="cuda", dtype=torch.float64, requires_grad=True)
    scene = builder(frequency)
    result = _solve(scene, components, "vjp", coherent=coherent)
    _loss(result).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()

    base = frequency.detach().clone()

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            frequency.copy_(value.to(device=frequency.device, dtype=frequency.dtype))
        try:
            return _loss(_solve(scene, components, "none", coherent=coherent)).detach()
        finally:
            with torch.no_grad():
                frequency.copy_(base)

    fd = central_difference_gradient(evaluate, base, _FREQ * FD_REL_STEP_FREQUENCY)
    assert relative_error(grad, fd, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def _tx_power_scattering_scene(power: torch.Tensor) -> Scene:
    # tx_power threads through the torch-side scattering NEE (source_power).
    wall = rough_wall_structure(
        2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=1.0,
        eps_r=4.0, sigma_e=0.05,
    )
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(
                position=torch.tensor(
                    [0.0, -1.0, 0.0], device=power.device
                ),
                power_w=power,
            ),
            make_receiver(
                position=torch.tensor([0.0, 1.0, 0.0], device=power.device)
            ),
        ],
        metadata={_FREQUENCY_METADATA_KEY: _FREQ},
    )


def _tx_power_same_side_scene(power: torch.Tensor) -> Scene:
    # A wall that does not occlude the tx->rx line. For a los-only solve tx_power
    # threads natively through the LoS endpoint-connection companion; for a
    # reflection-only solve it threads through the enumerated linear-coefficient
    # reattach.
    wall = same_side_wall_reflection_scene().structures[0]
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(
                position=torch.tensor(
                    [0.0, -1.0, 0.5], device=power.device
                ),
                power_w=power,
            ),
            make_receiver(
                position=torch.tensor([0.0, 1.0, 0.5], device=power.device)
            ),
        ],
        metadata={_FREQUENCY_METADATA_KEY: _FREQ},
    )


def _tx_power_endpoint_only_scene(power: torch.Tensor) -> Scene:
    # No structures + los-only routes through the endpoint-only fast path
    # (_build_endpoint_subpaths), which also threads tx_power natively under ad.
    return Scene(
        structures=[],
        endpoints=[
            make_transmitter(
                position=torch.tensor(
                    [0.0, 0.0, 0.0], device=power.device
                ),
                power_w=power,
            ),
            make_receiver(
                position=torch.tensor([5.0, 0.0, 0.0], device=power.device)
            ),
        ],
        metadata={_FREQUENCY_METADATA_KEY: _FREQ},
    )


@pytest.mark.parametrize(
    "builder,components",
    (
        # LoS companion (native grad_tx_power) with structures and via the
        # endpoint-only fast path, enumerated reflection block (linear-coefficient
        # reattach), and torch-side scattering NEE.
        (_tx_power_same_side_scene, frozenset({"los"})),
        (_tx_power_endpoint_only_scene, frozenset({"los"})),
        (_tx_power_same_side_scene, frozenset({"reflection"})),
        (_tx_power_scattering_scene, frozenset({"scattering"})),
    ),
)
def test_tx_power_grad_matches_fd(builder, components):
    _require_native()
    power = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
    scene = builder(power)
    result = _solve(scene, components, "vjp")
    _loss(result).backward()
    assert power.grad is not None, "tx_power must be a live leaf under ad != none"
    grad = power.grad.detach()

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        return _loss(_solve(builder(value.detach()), components, "none")).detach()

    fd = central_difference_gradient(
        evaluate, torch.tensor(2.0, dtype=torch.float64), 1.0e-3
    )
    assert relative_error(grad, fd, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# Gate 4: unbiased-gradient sanity (mean of per-seed grads == FD of the mean).
# ---------------------------------------------------------------------------


def test_seed_gradient_mean_matches_fd_of_mean():
    _require_native()
    scene = _transmission_scene()
    components = _COMPONENTS["transmission"]
    seeds = (1, 2, 3, 4)
    leaf = _material_leaf(scene, "layer_eps_r")

    grads = []
    leaf.requires_grad_(True)
    try:
        for seed in seeds:
            result = _solve(scene, components, "vjp", seed=seed)
            _loss(result).backward()
            grads.append(leaf.grad.detach().clone())
            leaf.grad = None
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    grad_mean = torch.stack(grads, dim=0).mean(dim=0)

    base = leaf.detach().clone()

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            leaf.copy_(values.to(device=leaf.device, dtype=leaf.dtype))
        try:
            total = torch.zeros((), device=leaf.device, dtype=torch.float32)
            for seed in seeds:
                total = total + _loss(_solve(scene, components, "none", seed=seed))
            return (total / len(seeds)).detach()
        finally:
            with torch.no_grad():
                leaf.copy_(base)

    fd_of_mean = central_difference_gradient(evaluate, base, FD_STEP_EPS_R)
    assert relative_error(grad_mean, fd_of_mean, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# Gate 5: geometry-gradient refusal through the stochastic sampler.
# ---------------------------------------------------------------------------


def test_shooting_metadata_reports_enumerated_blocks_only():
    _require_native()
    # A transmission-only fixture contributes solely through the shooting sampler,
    # whose hit-point geometry is frozen in v1 (ADR-022 stochastic-sampler stance).
    scene = _transmission_scene()
    result = _solve(scene, _COMPONENTS["transmission"], "vjp")
    assert result.metadata.get("ad_geometry") == "enumerated_blocks_only"


def test_shooting_geometry_gradient_is_refused_loudly():
    _require_native()
    # Requesting a mesh-vertex gradient that only reaches the stochastic sampler
    # must fail loudly (never a silent detach / zero). A live material leaf keeps
    # the output on the graph so the refusal, not a "does not require grad" error,
    # is what raises; the refusal may fire at solve time or at backward.
    def _geom_scene() -> tuple[Scene, torch.Tensor]:
        base_wall = transmission_wall_structure(
            3.0,
            PhysicalMaterial(
                layers=(MaterialLayer(thickness_m=0.08, eps_r=4.0, sigma_e=0.02),),
                name="bdpt-geom-wall",
            ),
        )
        vertex_leaf = (
            base_wall.geometry.to_mesh()[0]
            .clone()
            .to("cuda")
            .requires_grad_(True)
        )
        wall = make_mesh_structure(
            vertices=vertex_leaf,
            faces=torch.tensor(
                [[0, 1, 2], [1, 3, 2]], device=vertex_leaf.device
            ),
            material=PhysicalMaterial(
                layers=(MaterialLayer(thickness_m=0.08, eps_r=4.0, sigma_e=0.02),),
                name="bdpt-geom-wall",
            ),
            name="bdpt-geom-wall",
            surface_id=1,
        )
        scene = Scene(
            structures=[wall],
            endpoints=[
                make_transmitter(position=torch.tensor([0.0, 0.0, 0.0])),
                make_receiver(position=torch.tensor([6.0, 0.4, 0.2])),
            ],
            metadata={_FREQUENCY_METADATA_KEY: _FREQ},
        )
        return scene, vertex_leaf

    with pytest.raises((RuntimeError, NotImplementedError)):
        scene, vertex_leaf = _geom_scene()
        _compile(scene).materials.layer_eps_r.requires_grad_(True)
        result = _solve(scene, _COMPONENTS["transmission"], "vjp")
        _loss(result).backward()


# ---------------------------------------------------------------------------
# Gate 6: JVP-vs-VJP consistency at the solver level.
# ---------------------------------------------------------------------------


def test_jvp_vs_vjp_material_consistency():
    _require_native()
    scene = _reflection_scene()
    components = _COMPONENTS["reflection"]
    compiled = _compile(scene)
    base = compiled.materials.eps_r
    tangent = torch.ones_like(base)

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.detach().clone(), tangent)
        object.__setattr__(compiled.materials, "eps_r", dual)
        try:
            result = _solve(scene, components, "jvp")
            jvp = torch.autograd.forward_ad.unpack_dual(result.path_gain).tangent
        finally:
            object.__setattr__(compiled.materials, "eps_r", base)
    assert jvp is not None
    directional_forward = jvp.sum()

    base.requires_grad_(True)
    try:
        result = _solve(scene, components, "vjp")
        _loss(result).backward()
        directional_reverse = (base.grad.detach() * tangent).sum()
    finally:
        base.requires_grad_(False)
        base.grad = None
    assert relative_error(
        directional_forward, directional_reverse, abs_floor=ABS_TOL
    ) <= REL_TOL_GENERAL


def test_jvp_metadata_reports_dual_without_tape():
    _require_native()
    scene = _reflection_scene()
    result = _solve(scene, _COMPONENTS["reflection"], "jvp")
    kernel = result.metadata["kernel"]
    assert kernel["ad_status"] == "jvp"
    assert kernel["tape_bytes"] == 0