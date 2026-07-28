"""Solver-level AD through the native ADR-014 scattering companions.

End-to-end callers for the deterministic/path scattering component
(``component_id=6``): with ``ad_mode != "none"`` the ensemble and
realization-coherent scattering contributions must join the plan-07 AD system
so gradients reach frequency, the resident Kirchhoff BSDF table values, and the
phase-screen heights, and the total gradient stops silently dropping the
scattering component.

The wiring under test (``interactions/scattering.py`` building ``coef``
/ ``k0`` / ``amplitude_scale`` as Torch scalars and selecting the ``_ad``
wrappers when ``ad_mode != "none"``) is authored in parallel; these tests fail
loudly if it is missing or detaches a differentiable input. FD steps/tolerances
follow ``tests/ad/_tolerances.py`` and may need calibration once runnable.
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
from tests.support.scenes import rough_wall_structure
from witwin.core import Scene
from tests.support.core_world import make_receiver, make_transmitter
from witwin.channel.deployment import build_info
from witwin.core import PhaseScreen
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic import solve as deterministic_solve
from witwin.channel.path import Config as PathConfig
from witwin.channel.path import solve as path_solve
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_FREQUENCY_HZ = 3.0e9
# Directional FD steps for the resident-table and height gradients (both feed
# the kernels at O(0.1..1) table values / O(1e-3 m) heights).
_FD_STEP_TABLE = 1.0e-3
_FD_STEP_HEIGHT = 1.0e-4


def _require_rayd() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native scene capability is not built")


# ---------------------------------------------------------------------------
# Scene builders.
# ---------------------------------------------------------------------------


def _ensemble_scene() -> Scene:
    wall = rough_wall_structure(
        2.5, rms_height_m=0.015, corr_length_m=0.15, half_size=1.0
    )
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.0])),
        ],
    )


def _realization_screen(heights: torch.Tensor) -> PhaseScreen:
    return PhaseScreen(
        height=heights,
        height_scale_m=1.0,
        realization_id=1,
        mode="realization_coherent",
    )


def _realization_scene(heights: torch.Tensor) -> Scene:
    # Smooth material (scatter_model 0) + realization-coherent phase screen: the
    # screen replaces the specular lobe and drives scattering, so no Kirchhoff
    # ensemble table is built.
    wall = rough_wall_structure(
        2.5,
        rms_height_m=0.0,
        corr_length_m=0.15,
        half_size=1.0,
        phase_screen=_realization_screen(heights),
        with_uv=True,
    )
    return Scene(
        structures=[wall],
        endpoints=[
            make_transmitter(position=torch.tensor([0.0, -1.0, 0.0])),
            make_receiver(position=torch.tensor([0.0, 1.0, 0.0])),
        ],
    )


def _heights(seed: int = 5, size: int = 16) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (1.0e-3 * torch.randn(size, size, generator=generator)).to("cuda")


def _solve(
    scene: Scene,
    solver: str,
    ad_mode: str,
    *,
    reference_frequency_hz: float | torch.Tensor = _FREQUENCY_HZ,
):
    components = frozenset({"scattering"})
    if solver == "path":
        return path_solve(
            scene,
            PathConfig(
                max_depth=1,
                components=components,
                ad_mode=ad_mode,
                scattering_samples_per_m2=32.0,
            ),
            reference_frequency_hz=reference_frequency_hz,
        )
    return deterministic_solve(
        scene,
        DeterministicConfig(
            max_depth=1,
            components=components,
            ad_mode=ad_mode,
            scattering_samples_per_m2=32.0,
        ),
        reference_frequency_hz=reference_frequency_hz,
    )


def _loss(result) -> torch.Tensor:
    if hasattr(result, "component_power"):
        return result.component_power["scattering"].sum()
    # Path solver: every path in these scenes is a scattering path, so the
    # scattering component power is the incoherent sum of |a|^2 over valid paths.
    power = result.a.abs().square() * result.valid.unsqueeze(-1)
    return power.sum()


# ---------------------------------------------------------------------------
# (a) Ensemble scattering: frequency + resident table gradients.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["deterministic", "path"])
def test_ensemble_frequency_grad_matches_fd(solver):
    _require_rayd()
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = _ensemble_scene()
    result = _solve(scene, solver, "vjp", reference_frequency_hz=frequency)
    _loss(result).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_result = _solve(
            _ensemble_scene(),
            solver,
            "none",
            reference_frequency_hz=float(value),
        )
        return _loss(fd_result).detach()

    fd_grad = central_difference_gradient(
        evaluate,
        torch.tensor(_FREQUENCY_HZ, dtype=torch.float64),
        _FREQUENCY_HZ * FD_REL_STEP_FREQUENCY,
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_ensemble_table_value_grad_is_nonzero_and_fd_consistent():
    _require_rayd()
    scene = _ensemble_scene()
    compiled = compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ)
    stack = compiled.kirchhoff_resources.stack
    stack.f_te_flat.requires_grad_(True)
    baseline_f_te = stack.f_te_flat.detach().clone()

    result = _solve(scene, "deterministic", "vjp")
    _loss(result).backward()
    grad = stack.f_te_flat.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0
    grad = grad.detach()

    # Directional FD of the resident-table gradient on a fresh scene (the table
    # is a deterministic compile output, so the baseline point matches).
    generator = torch.Generator(device="cuda").manual_seed(7)
    direction = torch.randn(baseline_f_te.shape, generator=generator, device="cuda")

    def loss_with_table(table_te: torch.Tensor) -> torch.Tensor:
        fd_scene = _ensemble_scene()
        fd_stack = compile_scene(
            fd_scene,
            reference_frequency_hz=_FREQUENCY_HZ,
        ).kirchhoff_resources.stack
        with torch.no_grad():
            fd_stack.f_te_flat.copy_(table_te)
        return _loss(_solve(fd_scene, "deterministic", "none")).detach()

    fd_directional = central_difference_directional(
        loss_with_table, baseline_f_te, direction, _FD_STEP_TABLE
    )
    autograd_directional = (grad * direction).sum()
    assert (
        relative_error(autograd_directional, fd_directional, abs_floor=ABS_TOL)
        <= REL_TOL_GENERAL
    )


# ---------------------------------------------------------------------------
# (b) Realization scattering: phase-screen height + frequency gradients.
# ---------------------------------------------------------------------------


def test_realization_runtime_preserves_height_graph():
    # The autograd graph of a requires_grad height tensor must survive
    # PhaseScreenRuntime construction (ADR-014 wiring note). If it detaches,
    # this fails loudly instead of the height-gradient test silently zeroing.
    _require_rayd()
    heights = _heights().requires_grad_(True)
    scene = _realization_scene(heights)
    compiled = compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ)
    resources = compiled.phase_screen_resources.structures
    assert resources, "expected a compiled phase-screen resource"
    runtime = resources[min(resources)].runtime
    assert runtime.heights_m.requires_grad, "PhaseScreenRuntime detached heights"
    assert runtime.heights_m.grad_fn is not None


def test_realization_height_grad_is_nonzero_and_fd_consistent():
    _require_rayd()
    heights = _heights().requires_grad_(True)
    scene = _realization_scene(heights)
    result = _solve(scene, "deterministic", "vjp")
    _loss(result).backward()
    assert heights.grad is not None
    assert float(heights.grad.abs().sum()) > 0.0
    assert torch.isfinite(heights.grad).all()
    grad = heights.grad.detach()

    generator = torch.Generator(device="cuda").manual_seed(9)
    direction = torch.randn(heights.shape, generator=generator, device="cuda")

    def loss_with_heights(h: torch.Tensor) -> torch.Tensor:
        return _loss(_solve(_realization_scene(h), "deterministic", "none")).detach()

    fd_directional = central_difference_directional(
        loss_with_heights, heights.detach(), direction, _FD_STEP_HEIGHT
    )
    autograd_directional = (grad * direction).sum()
    assert (
        relative_error(autograd_directional, fd_directional, abs_floor=ABS_TOL)
        <= REL_TOL_GENERAL
    )


def test_realization_frequency_grad_matches_fd():
    _require_rayd()
    heights = _heights()
    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    scene = _realization_scene(heights)
    result = _solve(
        scene,
        "deterministic",
        "vjp",
        reference_frequency_hz=frequency,
    )
    _loss(result).backward()
    assert frequency.grad is not None
    grad = frequency.grad.detach()
    assert float(grad.abs()) > 0.0

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        fd_scene = _realization_scene(heights)
        return _loss(
            _solve(
                fd_scene,
                "deterministic",
                "none",
                reference_frequency_hz=float(value),
            )
        ).detach()

    fd_grad = central_difference_gradient(
        evaluate,
        torch.tensor(_FREQUENCY_HZ, dtype=torch.float64),
        _FREQUENCY_HZ * FD_REL_STEP_FREQUENCY,
    )
    assert relative_error(grad, fd_grad, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# (b') Realization scattering: layer-stack material gradients (ADR-015 Part B).
#
# The realization patch integral reads the smooth-stack Jones r_te/r_tm at the
# mean plane. Under ADR-015 Part B that stack becomes the differentiable
# em_layer_stack_ad, so the shared CSR layer leaves (the same tensors the MC
# transmission AD targets) carry gradients through the coherent realization
# solve.
# ---------------------------------------------------------------------------

_MATERIAL_FD_STEPS = {
    "layer_eps_r": FD_STEP_EPS_R,
    "layer_sigma_e": FD_STEP_SIGMA_E,
    "layer_thickness_m": FD_STEP_THICKNESS,
}


def _material_leaf(scene: Scene, name: str) -> torch.Tensor:
    return getattr(
        compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ).materials,
        name,
    )


def _fd_material_gradient(
    scene: Scene, leaf: torch.Tensor, step: float
) -> torch.Tensor:
    base = leaf.detach().clone()

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            leaf.copy_(values.to(device=leaf.device, dtype=leaf.dtype))
        try:
            return _loss(_solve(scene, "deterministic", "none")).detach()
        finally:
            with torch.no_grad():
                leaf.copy_(base)

    return central_difference_gradient(evaluate, base, step)


@pytest.mark.parametrize("param", ("layer_eps_r", "layer_sigma_e", "layer_thickness_m"))
def test_realization_material_grad_is_nonzero_and_fd_consistent(param):
    _require_rayd()
    scene = _realization_scene(_heights())
    leaf = _material_leaf(scene, param)
    expected = _fd_material_gradient(scene, leaf, _MATERIAL_FD_STEPS[param])

    leaf.requires_grad_(True)
    try:
        result = _solve(scene, "deterministic", "vjp")
        loss = _loss(result)
        assert loss.requires_grad, "realization material leaf detached from the loss"
        loss.backward()
        assert leaf.grad is not None
        assert float(leaf.grad.abs().sum()) > 0.0
        assert torch.isfinite(leaf.grad).all()
        assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


def test_realization_material_jvp_matches_vjp():
    # Forward/reverse duality of the layer_eps_r gradient through the coherent
    # realization patch integral: the JVP (em_layer_stack_ad forward mode) and
    # the VJP contraction on the same tangent must agree.
    _require_rayd()
    scene = _realization_scene(_heights())
    leaf = _material_leaf(scene, "layer_eps_r")
    generator = torch.Generator(device="cpu").manual_seed(41)
    tangent = torch.randn(leaf.shape, generator=generator).to(leaf.device)

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(leaf.detach(), tangent)
        compiled = compile_scene(scene, reference_frequency_hz=_FREQUENCY_HZ)
        object.__setattr__(compiled.materials, "layer_eps_r", dual)
        try:
            jvp = torch.autograd.forward_ad.unpack_dual(
                _loss(_solve(scene, "deterministic", "jvp"))
            ).tangent
        finally:
            object.__setattr__(compiled.materials, "layer_eps_r", leaf)
    assert jvp is not None

    leaf.requires_grad_(True)
    try:
        _loss(_solve(scene, "deterministic", "vjp")).backward()
        vjp = (leaf.grad * tangent).sum()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_realization_material_ad_mode_none_builds_no_graph():
    # A requires_grad material leaf must not create a graph when ad is off: the
    # plain em_layer_stack_eval path stays detached and bitwise-primal.
    _require_rayd()
    scene = _realization_scene(_heights())
    leaf = _material_leaf(scene, "layer_eps_r")
    leaf.requires_grad_(True)
    try:
        power = _solve(scene, "deterministic", "none").component_power["scattering"]
        assert not power.requires_grad
        assert torch.autograd.forward_ad.unpack_dual(power).tangent is None
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


# ---------------------------------------------------------------------------
# (c) JVP path consistency versus VJP.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scene_kind", ["ensemble", "realization"])
def test_scattering_frequency_jvp_matches_vjp(scene_kind):
    _require_rayd()
    heights = _heights()

    def build():
        if scene_kind == "ensemble":
            return _ensemble_scene()
        return _realization_scene(heights)

    frequency = torch.tensor(
        _FREQUENCY_HZ, dtype=torch.float64, device="cuda", requires_grad=True
    )
    result = _solve(
        build(),
        "deterministic",
        "vjp",
        reference_frequency_hz=frequency,
    )
    _loss(result).backward()
    vjp_grad = float(frequency.grad.detach())

    primal = torch.tensor(_FREQUENCY_HZ, dtype=torch.float64, device="cuda")
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            primal.clone(), torch.ones_like(primal)
        )
        dual_result = _solve(
            build(),
            "deterministic",
            "jvp",
            reference_frequency_hz=dual,
        )
        tangent = torch.autograd.forward_ad.unpack_dual(
            dual_result.component_power["scattering"]
        ).tangent
    assert tangent is not None
    jvp_value = float(tangent.sum())
    assert jvp_value == pytest.approx(vjp_grad, rel=1.0e-3, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# (d) ad_mode="none" stays primal (no autograd.Function, no requires_grad).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scene_kind", ["ensemble", "realization"])
def test_scattering_ad_mode_none_stays_primal(scene_kind):
    _require_rayd()
    if scene_kind == "ensemble":
        scene = _ensemble_scene()
    else:
        scene = _realization_scene(_heights())
    result = _solve(scene, "deterministic", "none")
    power = result.component_power["scattering"]
    assert not power.requires_grad
    assert torch.autograd.forward_ad.unpack_dual(power).tangent is None


# ---------------------------------------------------------------------------
# (e) Fixed-input rejection at the AD wrapper level.
# ---------------------------------------------------------------------------


def test_ensemble_wrapper_rejects_fixed_receiver_polarization():
    # rx_pol is a fixed ADR-014 op-1 input; requesting its gradient through the
    # public wrapper must fail loudly (no silent detach, no wrong gradient).
    from witwin.channel.kernels import scattering as scattering_autograd

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(404)

    def randn(*shape):
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float32
        )

    samples, rows, num_rx = 4, 6, 2
    n_o = torch.nn.functional.normalize(randn(samples, 3), dim=-1)
    t1r = torch.nn.functional.normalize(
        torch.cross(n_o, randn(samples, 3), dim=-1), dim=-1
    )
    t2r = torch.cross(n_o, t1r, dim=-1)
    wi_local = torch.stack(
        (
            randn(samples),
            randn(samples),
            0.5 + 0.3 * torch.rand(samples, generator=generator, device=device),
        ),
        dim=-1,
    )
    rx_pol = torch.nn.functional.normalize(randn(num_rx, 3), dim=-1).requires_grad_(
        True
    )
    nti = nto = npo = 8
    f_te = 0.2 + torch.rand(nti, 1, nto, npo, generator=generator, device=device)
    f_tm = 0.2 + torch.rand(nti, 1, nto, npo, generator=generator, device=device)
    coef = torch.tensor(0.3, dtype=torch.float32, device=device)

    out = scattering_autograd.scattering_ensemble_eval_ad(
        torch.ones(rows, dtype=torch.bool, device=device),
        torch.nn.functional.normalize(randn(rows, 3), dim=-1),  # wo_rows
        0.5 + torch.rand(rows, generator=generator, device=device),  # r2_rows
        0.3 + 0.5 * torch.rand(rows, generator=generator, device=device),  # cos_o_rows
        n_o,
        t1r,
        t2r,
        wi_local,
        0.3 + 0.5 * torch.rand(samples, generator=generator, device=device),  # cos_i
        0.5 + torch.rand(samples, generator=generator, device=device),  # r1
        0.2 + torch.rand(samples, generator=generator, device=device),  # a_te2
        0.2 + torch.rand(samples, generator=generator, device=device),  # a_tm2
        0.2 + torch.rand(samples, generator=generator, device=device),  # weights
        torch.zeros(samples, dtype=torch.int32, device=device),  # material_id
        t1r,  # backup_axis
        rx_pol,
        torch.randint(
            0, num_rx, (rows,), generator=generator, device=device, dtype=torch.int64
        ),  # rc_idx
        torch.randint(
            0, samples, (rows,), generator=generator, device=device, dtype=torch.int64
        ),  # sc_idx
        f_te.reshape(-1).contiguous(),
        f_tm.reshape(-1).contiguous(),
        torch.zeros(1, dtype=torch.int64, device=device),  # table_offset
        torch.tensor(
            [[nti, 1, nto, npo]], dtype=torch.int32, device=device
        ),  # table_dims
        torch.zeros(1, dtype=torch.int32, device=device),  # material_slot
        coef=coef,
        threshold=-1.0,
    )
    with pytest.raises(NotImplementedError):
        out["gain"].sum().backward()
