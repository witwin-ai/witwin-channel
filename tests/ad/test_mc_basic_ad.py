"""AD-3 solver-level tests: montecarlo.basic real power-map gradients.

The M column of the plan 07 section 9.3 matrix. Unlike deterministic/path,
montecarlo.basic emits an incoherent REAL power map, so these check the power
gain rather than a complex coefficient.

Two contracts this file pins, both of which the implementation has to honor:

- Materials come from the compiled material store in BOTH ad_mode="none" and
  the AD modes. The solver used to flatten materials to host floats
  (_host_material_tensors -> bdpt_face_material_tensors_from_host), which
  cannot carry a gradient and, worse, would make a finite difference taken on
  the store measure exactly zero. One material source, same values.
- Receiver-position gradients only exist for point receivers. A ReceiverGrid is
  generated natively from origin/axes/spacing and exposes no per-receiver
  position leaf, and for a radiomap the grid is the output, not a parameter.

The reflection/diffraction maps bin contributions into grid cells by hit
position, so a moving transmitter changes cell assignment discretely. Those
assignments are part of the frozen winner (plan 07 section 4): the gradient
describes the continuous part only. For the Sionna-style reflection radiomap
the continuous part with respect to the transmitter is IDENTICALLY ZERO: the
per-ray deposit weight is |Gamma|^2 * solid_angle * (lambda/4pi)^2 /
(A_cell * |cos|), whose factors depend only on the frozen sampled direction,
the face normal and the materials; the 1/d^2 spreading lives in the frozen
ray density and the binning. Measured on this scene at seed 7, the primal
map total is piecewise constant in tx: central differences read single-ray
binning jumps at h = 1e-2 (about 7e-7, one deposit's worth per 2h) and only
float32 reassociation noise (about 6e-11) at h <= 1e-3, with no continuous
component at any step. The transmitter test below therefore pins the exact
analytic contract (zero VJP gradient and zero JVP tangent through a live
graph) instead of comparing against sampling-jump noise.
"""

from __future__ import annotations

import pytest
import torch

from tests.ad._fd import central_difference_gradient, relative_error
from tests.ad._tolerances import (
    ABS_TOL,
    FD_REL_STEP_FREQUENCY,
    FD_STEP_EPS_R,
    FD_STEP_POSITION,
    FD_STEP_SIGMA_E,
    FD_STEP_THICKNESS,
    REL_TOL_GENERAL,
)
from tests.support.scenes import transmission_wall_structure
from witwin.channel_native import (
    ReceiverGrid,
    ReceiverPoint,
    Scene,
    Structure,
    Transmitter,
)
from witwin.channel_native.core.materials import Dielectric, Layer, PhysicalSurface
from witwin.channel_native.montecarlo.basic import Config, solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_FREQUENCY_HZ = 3.0e9
_SAMPLES = 4096
_SEED = 7
_TX = (0.0, -1.0, 0.5)


def _grid() -> ReceiverGrid:
    return ReceiverGrid(
        origin=torch.tensor([1.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _wall() -> Structure:
    return Structure(
        vertices=torch.tensor(
            [
                [2.5, -3.0, -1.0],
                [2.5, 3.0, -1.0],
                [2.5, -3.0, 2.0],
                [2.5, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=Dielectric(eps_r=4.0, sigma_e=0.02),
        name="ad-mc-wall",
        surface_id=1,
    )


def _reflection_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
) -> Scene:
    position = torch.tensor(_TX) if tx is None else tx
    return Scene(
        structures=[_wall()],
        transmitters=[Transmitter(position=position)],
        receivers=[_grid()],
        frequency=frequency,
    )


def _transmission_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
) -> Scene:
    material = PhysicalSurface(
        layers=(
            Layer(thickness_m=0.06, eps_r=4.0, sigma_e=0.02),
            Layer(thickness_m=0.09, eps_r=2.5, sigma_e=0.01),
        ),
        name="ad-mc-sheet",
    )
    position = torch.tensor([0.0, 0.0, 0.0]) if tx is None else tx
    return Scene(
        structures=[transmission_wall_structure(3.0, material)],
        transmitters=[Transmitter(position=position)],
        receivers=[
            ReceiverGrid(
                origin=torch.tensor([6.0, -1.0, -0.5]),
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(4, 4),
                spacing=(0.5, 0.25),
            )
        ],
        frequency=frequency,
    )


def _los_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
) -> Scene:
    return Scene(
        structures=[],
        transmitters=[Transmitter(position=torch.tensor(_TX) if tx is None else tx)],
        receivers=[
            ReceiverPoint(
                position=torch.tensor([0.0, 1.0, 0.5]) if rx is None else rx
            )
        ],
        frequency=frequency,
    )


def _solve(scene: Scene, components: frozenset[str], ad_mode: str):
    return solve(
        scene,
        Config(
            samples=_SAMPLES,
            seed=_SEED,
            components=set(components),
            max_depth=1,
            ad_mode=ad_mode,
        ),
    )


def _loss(result) -> torch.Tensor:
    return result.path_gain.sum()


def _material_leaf(scene: Scene, name: str) -> torch.Tensor:
    return getattr(scene.compile().materials, name)


_MATERIAL_STEPS = {
    "eps_r": FD_STEP_EPS_R,
    "sigma_e": FD_STEP_SIGMA_E,
    "layer_eps_r": FD_STEP_EPS_R,
    "layer_sigma_e": FD_STEP_SIGMA_E,
    "layer_thickness_m": FD_STEP_THICKNESS,
}


def _fd_material_gradient(
    scene: Scene, components: frozenset[str], leaf: torch.Tensor, step: float
) -> torch.Tensor:
    base = leaf.detach().clone()

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            leaf.copy_(values.to(device=leaf.device, dtype=leaf.dtype))
        try:
            return _loss(_solve(scene, components, "none")).detach()
        finally:
            with torch.no_grad():
                leaf.copy_(base)

    return central_difference_gradient(evaluate, base, step)


@pytest.mark.parametrize("param", ("eps_r", "sigma_e"))
def test_reflection_material_grad_matches_fd(param):
    scene = _reflection_scene()
    components = frozenset({"reflection"})
    leaf = _material_leaf(scene, param)
    expected = _fd_material_gradient(scene, components, leaf, _MATERIAL_STEPS[param])

    leaf.requires_grad_(True)
    try:
        _loss(_solve(scene, components, "vjp")).backward()
        assert leaf.grad is not None
        assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


@pytest.mark.parametrize(
    "param", ("layer_eps_r", "layer_sigma_e", "layer_thickness_m")
)
def test_transmission_material_grad_matches_fd(param):
    scene = _transmission_scene()
    components = frozenset({"los", "transmission"})
    leaf = _material_leaf(scene, param)
    expected = _fd_material_gradient(scene, components, leaf, _MATERIAL_STEPS[param])

    leaf.requires_grad_(True)
    try:
        _loss(_solve(scene, components, "vjp")).backward()
        assert leaf.grad is not None
        assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


@pytest.mark.parametrize(
    "builder,components",
    (
        (_los_scene, frozenset({"los"})),
        (_reflection_scene, frozenset({"reflection"})),
        (_transmission_scene, frozenset({"los", "transmission"})),
    ),
)
def test_frequency_grad_matches_fd(builder, components):
    step = FD_REL_STEP_FREQUENCY * _FREQUENCY_HZ

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        scene = builder(frequency=float(value))
        return _loss(_solve(scene, components, "none")).detach()

    expected = central_difference_gradient(
        evaluate, torch.tensor(_FREQUENCY_HZ), step
    )

    frequency = torch.tensor(_FREQUENCY_HZ, device="cuda").requires_grad_(True)
    scene = builder(frequency=frequency)
    _loss(_solve(scene, components, "vjp")).backward()
    assert frequency.grad is not None
    assert (
        relative_error(frequency.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


@pytest.mark.parametrize("endpoint", ("tx", "rx"))
def test_los_endpoint_position_grad_matches_fd(endpoint):
    components = frozenset({"los"})
    base = torch.tensor(_TX if endpoint == "tx" else (0.0, 1.0, 0.5))

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        scene = _los_scene(**{endpoint: values.clone()})
        return _loss(_solve(scene, components, "none")).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_POSITION)

    leaf = base.clone().cuda().requires_grad_(True)
    scene = _los_scene(**{endpoint: leaf})
    _loss(_solve(scene, components, "vjp")).backward()
    assert leaf.grad is not None
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_reflection_transmitter_position_grad_is_the_exact_continuous_part():
    """TX position on a binned power map: frozen cell assignment, continuous part only.

    The continuous part is identically zero for this estimator (module
    docstring): the deposit weight carries no ray-origin dependence, and the
    origin's only influence (which cell, whether the deposit lands) is a
    frozen discrete winner. A finite difference here measures nothing but
    single-ray binning jumps (measured 7e-7-scale at the h = 1e-2 spec step,
    float32 reassociation noise below h = 1e-3), so the assertion pins the
    analytic contract deterministically in BOTH modes: backward must reach
    the transmitter leaf and deliver the exact zero, and a forward-mode
    transmitter tangent must produce an exactly zero loss tangent.
    """

    components = frozenset({"reflection"})
    base = torch.tensor(_TX)

    leaf = base.clone().cuda().requires_grad_(True)
    scene = _reflection_scene(tx=leaf)
    loss = _loss(_solve(scene, components, "vjp"))
    assert loss.requires_grad
    loss.backward()
    assert leaf.grad is not None
    assert torch.all(leaf.grad == 0.0)

    tangent = torch.tensor([1.0, -2.0, 0.5], device="cuda")
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.clone().cuda(), tangent)
        scene = _reflection_scene(tx=dual)
        jvp = torch.autograd.forward_ad.unpack_dual(
            _loss(_solve(scene, components, "jvp"))
        ).tangent
    assert jvp is None or float(jvp) == 0.0


def test_forward_mode_material_dual_matches_reverse():
    scene = _reflection_scene()
    components = frozenset({"reflection"})
    leaf = _material_leaf(scene, "eps_r")
    generator = torch.Generator(device="cpu").manual_seed(23)
    tangent = torch.randn(leaf.shape, generator=generator).to(leaf.device)

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(leaf.detach(), tangent)
        object.__setattr__(scene.compile().materials, "eps_r", dual)
        try:
            jvp = torch.autograd.forward_ad.unpack_dual(
                _loss(_solve(scene, components, "jvp"))
            ).tangent
        finally:
            object.__setattr__(scene.compile().materials, "eps_r", leaf)
    assert jvp is not None

    leaf.requires_grad_(True)
    try:
        _loss(_solve(scene, components, "vjp")).backward()
        vjp = (leaf.grad * tangent).sum()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_transmission_forward_mode_layer_dual_matches_reverse():
    """Forward/reverse duality through the layer stack, LoS and finalize glue."""

    scene = _transmission_scene()
    components = frozenset({"los", "transmission"})
    leaf = _material_leaf(scene, "layer_eps_r")
    generator = torch.Generator(device="cpu").manual_seed(31)
    tangent = torch.randn(leaf.shape, generator=generator).to(leaf.device)

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(leaf.detach(), tangent)
        object.__setattr__(scene.compile().materials, "layer_eps_r", dual)
        try:
            jvp = torch.autograd.forward_ad.unpack_dual(
                _loss(_solve(scene, components, "jvp"))
            ).tangent
        finally:
            object.__setattr__(scene.compile().materials, "layer_eps_r", leaf)
    assert jvp is not None

    leaf.requires_grad_(True)
    try:
        _loss(_solve(scene, components, "vjp")).backward()
        vjp = (leaf.grad * tangent).sum()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_gradient_is_stable_across_seeds():
    """MC gradients are random: the spread across seeds must stay bounded.

    The plan 07 section 9.4 cross-seed check the original channel never had.
    Reflection sampling is seed-independent by construction, so this exercises
    the seeded component (diffraction) against its own sampling noise.
    """

    scene = _reflection_scene()
    components = frozenset({"reflection"})
    leaf = _material_leaf(scene, "eps_r")
    gradients = []
    leaf.requires_grad_(True)
    try:
        for seed in (1, 2, 3, 4, 5):
            result = solve(
                scene,
                Config(
                    samples=_SAMPLES,
                    seed=seed,
                    components=set(components),
                    max_depth=1,
                    ad_mode="vjp",
                ),
            )
            _loss(result).backward()
            gradients.append(leaf.grad.detach().clone())
            leaf.grad = None
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None

    stacked = torch.stack(gradients)
    mean = stacked.mean(dim=0)
    spread = stacked.std(dim=0)
    scale = mean.abs().clamp_min(ABS_TOL)
    assert float((spread / scale).max()) <= 0.1


def test_ad_mode_none_keeps_primal_contract():
    """The AD-mode forward is the primal computation, not a reimplementation.

    The AD path runs the exact same native kernels (accumulate, layout,
    finalize) behind dispatch-only autograd Functions, so the only admissible
    difference between two solves is the reflection kernel's own atomicAdd
    scheduling: measured on this scene, two ad_mode="none" solves of the same
    Scene object already differ in a few cells by up to two float32 ulp
    (about 1e-7 relative), and none-vs-vjp shows the same envelope with no
    systematic offset. The comparison therefore allows that measured
    reordering envelope and nothing more; cells with no deposits are exactly
    zero in both modes (the deposit SET is deterministic, only the intra-cell
    summation order floats), which atol=0 pins.
    """

    scene = _reflection_scene()
    components = frozenset({"reflection"})
    leaf = _material_leaf(scene, "eps_r")
    leaf.requires_grad_(True)
    try:
        primal = _solve(scene, components, "none")
        ad = _solve(scene, components, "vjp")
    finally:
        leaf.requires_grad_(False)

    assert not primal.path_gain.requires_grad
    assert primal.path_gain.grad_fn is None
    assert ad.path_gain.requires_grad
    torch.testing.assert_close(
        primal.path_gain, ad.path_gain.detach(), rtol=5.0e-7, atol=0.0
    )
    assert primal.metadata["kernel"]["ad_status"] == "none"
    assert ad.metadata["kernel"]["ad_status"] == "vjp"


def test_scattering_component_fails_loudly_in_ad_mode():
    scene = _reflection_scene()
    with pytest.raises((RuntimeError, NotImplementedError), match="scattering"):
        _solve(scene, frozenset({"reflection", "scattering"}), "vjp")
