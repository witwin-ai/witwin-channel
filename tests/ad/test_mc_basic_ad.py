# Copyright Xingyu Chen.
# AD solver-level tests: montecarlo.basic real power-map gradients.

"""AD solver-level tests: montecarlo.basic real power-map gradients."""

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
from witwin.core import ReceiverGrid, Scene, Structure
from tests.support.core_world import (
    make_mesh_structure,
    make_receiver,
    make_receiver_grid,
    make_transmitter,
)
from witwin.core import MaterialLayer, PhysicalMaterial
from witwin.channel.montecarlo.basic import Config, solve
from witwin.channel.scene import compile as compile_scene

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_FREQUENCY_HZ = 3.0e9
_SAMPLES = 4096
_SEED = 7
_TX = (0.0, -1.0, 0.5)
_FREQUENCY_METADATA_KEY = "test_reference_frequency_hz"


def _grid() -> ReceiverGrid:
    return make_receiver_grid(
        origin=torch.tensor([1.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _wall() -> Structure:
    return make_mesh_structure(
        vertices=torch.tensor(
            [
                [2.5, -3.0, -1.0],
                [2.5, 3.0, -1.0],
                [2.5, -3.0, 2.0],
                [2.5, 3.0, 2.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalMaterial(eps_r=4.0, sigma_e=0.02),
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
        endpoints=[make_transmitter(position=position), _grid()],
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _transmission_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
) -> Scene:
    material = PhysicalMaterial(
        layers=(
            MaterialLayer(thickness_m=0.06, eps_r=4.0, sigma_e=0.02),
            MaterialLayer(thickness_m=0.09, eps_r=2.5, sigma_e=0.01),
        ),
        name="ad-mc-sheet",
    )
    position = torch.tensor([0.0, 0.0, 0.0]) if tx is None else tx
    return Scene(
        structures=[transmission_wall_structure(3.0, material)],
        endpoints=[
            make_transmitter(position=position),
            make_receiver_grid(
                origin=torch.tensor([6.0, -1.0, -0.5]),
                x_axis=torch.tensor([0.0, 1.0, 0.0]),
                y_axis=torch.tensor([0.0, 0.0, 1.0]),
                shape=(4, 4),
                spacing=(0.5, 0.25),
            )
        ],
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _los_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
    rx: torch.Tensor | None = None,
) -> Scene:
    return Scene(
        structures=[],
        endpoints=[
            make_transmitter(
                position=torch.tensor(_TX) if tx is None else tx
            ),
            make_receiver(
                position=torch.tensor([0.0, 1.0, 0.5]) if rx is None else rx
            )
        ],
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _reference_frequency(scene: Scene):
    return scene.metadata[_FREQUENCY_METADATA_KEY]


def _compile(scene: Scene):
    return compile_scene(
        scene,
        reference_frequency_hz=_reference_frequency(scene),
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
        reference_frequency_hz=_reference_frequency(scene),
    )


def _loss(result) -> torch.Tensor:
    return result.path_gain.sum()


def _material_leaf(scene: Scene, name: str) -> torch.Tensor:
    return getattr(_compile(scene).materials, name)


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


def test_transmission_transmitter_position_grad_matches_fd():
    """TX position through the transmission radiomap (the derivative capability matrix).

 Both factors of the map move with the transmitter: the analytic per-cell
 Friis matrix (through the LoS Function) and every per-wall power
 transmittance (the straight-line incidence cosine is a function of the
 live march origin). The wall-crossing set is the frozen winner and stays
 stable across the FD probes on this fixture.
 """

    components = frozenset({"los", "transmission"})
    base = torch.tensor([0.0, 0.0, 0.0])

    def evaluate(values: torch.Tensor) -> torch.Tensor:
        scene = _transmission_scene(tx=values.clone())
        return _loss(_solve(scene, components, "none")).detach()

    expected = central_difference_gradient(evaluate, base, FD_STEP_POSITION)

    leaf = base.clone().cuda().requires_grad_(True)
    scene = _transmission_scene(tx=leaf)
    _loss(_solve(scene, components, "vjp")).backward()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().max()) > 0.0
    assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


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
        object.__setattr__(_compile(scene).materials, "eps_r", dual)
        try:
            jvp = torch.autograd.forward_ad.unpack_dual(
                _loss(_solve(scene, components, "jvp"))
            ).tangent
        finally:
            object.__setattr__(_compile(scene).materials, "eps_r", leaf)
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
        object.__setattr__(_compile(scene).materials, "layer_eps_r", dual)
        try:
            jvp = torch.autograd.forward_ad.unpack_dual(
                _loss(_solve(scene, components, "jvp"))
            ).tangent
        finally:
            object.__setattr__(_compile(scene).materials, "layer_eps_r", leaf)
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

 the AD cross-seed check the original channel never had.
 Reflection sampling is seed-independent by construction (deterministic
 Fibonacci directions), so its cross-seed spread is a numerical floor;
 the genuinely seeded component (diffraction) gets its own check below.
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
                reference_frequency_hz=_reference_frequency(scene),
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


# ---------------------------------------------------------------------------
# Diffraction map (diffraction AD): the M x diffraction column. The map is
# assembled from RayD's frozen Keller-cone sampling tape; the differentiable
# inputs are the wedge-face slab materials, the carrier frequency and the
# transmitter position (the incident spherical wave, the incidence angles and
# the recomputed plane-crossing Jacobian all move continuously with the
# source under the frozen tape).
# ---------------------------------------------------------------------------

_DIFFRACTION_SAMPLES = 4096


def _diffraction_scene(
    frequency: float | torch.Tensor = _FREQUENCY_HZ,
    tx: torch.Tensor | None = None,
) -> Scene:
    from tests.support.scenes import wedge_diffraction_scene

    base = wedge_diffraction_scene(
        PhysicalMaterial(eps_r=4.0, sigma_e=0.02),
        tx=tx,
    )
    transmitter = next(
        endpoint for endpoint in base.endpoints if endpoint.role == "tx"
    )
    grid = make_receiver_grid(
            origin=torch.tensor([3.0, -1.0, -0.5]),
            x_axis=torch.tensor([0.0, 1.0, 0.0]),
            y_axis=torch.tensor([0.0, 0.0, 1.0]),
            shape=(4, 4),
            spacing=(2.0 / 3.0, 1.0 / 3.0),
    )
    return Scene(
        structures=base.structures,
        endpoints=[transmitter, grid],
        metadata={_FREQUENCY_METADATA_KEY: frequency},
    )


def _solve_diffraction(scene: Scene, ad_mode: str, seed: int = _SEED):
    return solve(
        scene,
        Config(
            samples=_DIFFRACTION_SAMPLES,
            seed=seed,
            components={"diffraction"},
            max_depth=1,
            ad_mode=ad_mode,
        ),
        reference_frequency_hz=_reference_frequency(scene),
    )


def test_diffraction_ad_forward_is_the_primal_map():
    """The AD forward dispatches the exact primal tape-accumulate kernel."""

    scene = _diffraction_scene()
    leaf = _material_leaf(scene, "eps_r")
    leaf.requires_grad_(True)
    try:
        primal = _solve_diffraction(scene, "none")
        ad = _solve_diffraction(scene, "vjp")
    finally:
        leaf.requires_grad_(False)
    assert float(primal.path_gain.abs().sum()) > 0.0
    assert ad.path_gain.requires_grad
    torch.testing.assert_close(
        primal.path_gain, ad.path_gain.detach(), rtol=5.0e-7, atol=0.0
    )


@pytest.mark.parametrize("param", ("eps_r", "sigma_e"))
def test_diffraction_material_grad_matches_fd(param):
    scene = _diffraction_scene()
    leaf = _material_leaf(scene, param)
    expected = _fd_material_gradient(
        scene, frozenset({"diffraction"}), leaf, _MATERIAL_STEPS[param]
    )

    leaf.requires_grad_(True)
    try:
        _loss(_solve_diffraction(scene, "vjp")).backward()
        assert leaf.grad is not None
        assert float(leaf.grad.abs().max()) > 0.0
        assert relative_error(leaf.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None


def test_diffraction_frequency_grad_matches_fd():
    step = FD_REL_STEP_FREQUENCY * _FREQUENCY_HZ

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        scene = _diffraction_scene(frequency=float(value))
        return _loss(_solve_diffraction(scene, "none")).detach()

    expected = central_difference_gradient(evaluate, torch.tensor(_FREQUENCY_HZ), step)

    frequency = torch.tensor(_FREQUENCY_HZ, device="cuda").requires_grad_(True)
    scene = _diffraction_scene(frequency=frequency)
    _loss(_solve_diffraction(scene, "vjp")).backward()
    assert frequency.grad is not None
    assert float(frequency.grad.abs()) > 0.0
    assert (
        relative_error(frequency.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL
    )


def _captured_diffraction_tape(monkeypatch, tx: torch.Tensor) -> tuple:
    """Solve once in vjp mode and capture the tape-accumulate Function args."""

    from witwin.channel.kernels import montecarlo as _maps

    captured: dict[str, tuple] = {}
    original = _maps._McDiffractionMapAdFunction.forward

    def spy(*args):
        captured["args"] = args
        return original(*args)

    monkeypatch.setattr(
        _maps._McDiffractionMapAdFunction, "forward", staticmethod(spy)
    )
    leaf = tx.clone().cuda().requires_grad_(True)
    scene = _diffraction_scene(tx=leaf)
    _loss(_solve_diffraction(scene, "vjp")).backward()
    monkeypatch.undo()
    assert leaf.grad is not None
    assert float(leaf.grad.abs().max()) > 0.0
    assert bool(torch.isfinite(leaf.grad).all())
    return captured["args"]


def test_diffraction_transmitter_position_grad_matches_fixed_tape_fd(monkeypatch):
    """TX position through the diffraction map: fixed-tape FD parity.

 Unlike the reflection radiomap (whose deposit weight is analytically
 origin-independent), the diffraction deposit carries the incident
 spherical wave from the source, so its continuous source derivative is
 genuinely nonzero. A FULL-SOLVE central difference is NOT a valid oracle
 for this cell, measured on this fixture: moving the transmitter churns
 the discrete winners at every step size (the wedge-event discovery rays,
 the RayD sampling tape, and the per-lane Keller-cone acceptance test
 `phi <= exterior_angle` whose phi rotates with the source), so the FD
 never converges in h (probed h = 3e-4 .. 3e-2, both full-solve and
 frozen-tape map sums; sign flips throughout). The valid oracle is the
 fixed-winner contract itself: freeze the tape, keep only lanes whose
 accept decisions survive the /-h probes, and compare the aggregated
 per-lane central differences against the aggregated dual. Aggregation
 over many lanes averages away the per-lane float32 deposit noise and
 the frozen pseudo-infinite truncation-factor ripple that ride on the
 single-lane FD (measured ~10-15% per lane at the converged step).
 """

    from witwin.channel.kernels import montecarlo as _ops

    base = torch.tensor([0.0, -1.0, 0.5])
    args = _captured_diffraction_tape(monkeypatch, base)
    params = args[23]
    tape_active, tape_state, tape_cell, tape_u = args[6:10]
    states = list(args[10:21])
    material_mu_r, material_valid = args[21], args[22]
    materials = [args[index].detach() for index in (1, 2, 3, 4)]
    kernel_kwargs = {
        "tx_pol": params["tx_pol"],
        "grid_axis": params["grid_axis"],
        "grid_position": params["grid_position"],
        "grid_resolution0": params["grid_resolution0"],
        "grid_resolution1": params["grid_resolution1"],
        "wavelength": params["wavelength"],
        "grid_cell_area": params["grid_cell_area"],
        "seed": params["seed"],
        "total_edge_length": params["total_edge_length"],
    }

    def one_lane_sum(lane: int, src_rows: torch.Tensor) -> float:
        mask = torch.zeros_like(tape_active)
        mask[lane] = tape_active[lane]
        rows = list(states)
        rows[9] = src_rows
        return float(
            _ops.mc_utd_diffraction_tape_accumulate(
                mask,
                tape_state,
                tape_cell,
                tape_u,
                *rows,
                materials[0],
                materials[1],
                material_mu_r,
                materials[2],
                material_valid,
                materials[3],
                params["grid_axis"],
                params["grid_position"],
                params["grid_coord0_min"],
                params["grid_coord0_max"],
                params["grid_coord1_min"],
                params["grid_coord1_max"],
                params["grid_resolution0"],
                params["grid_resolution1"],
                params["wavelength"],
                params["grid_cell_area"],
                params["seed"],
                params["total_edge_length"],
                params["tx_pol"],
            ).sum()
        )

    def lane_dual(lane: int) -> torch.Tensor:
        mask = torch.zeros_like(tape_active)
        mask[lane] = True
        dual = torch.zeros(3)
        for axis in range(3):
            tangent_src = torch.zeros(3, device="cuda")
            tangent_src[axis] = 1.0
            tangent = _ops.mc_utd_diffraction_tape_accumulate_jvp(
                (mask, tape_state, tape_cell, tape_u),
                tuple(states),
                materials[0],
                materials[1],
                material_mu_r,
                materials[2],
                material_valid,
                materials[3],
                None,
                None,
                None,
                None,
                tangent_src,
                wavelength_tangent=0.0,
                **kernel_kwargs,
            )
            dual[axis] = float(tangent.sum())
        return dual

    src0 = states[9].detach()
    active_indices = torch.nonzero(tape_active).flatten().tolist()

    def lane_fd(lane: int, step: float) -> torch.Tensor | None:
        fd = torch.zeros(3)
        for axis in range(3):
            plus = src0.clone()
            minus = src0.clone()
            plus[:, axis] += step
            minus[:, axis] -= step
            f_plus = one_lane_sum(lane, plus)
            f_minus = one_lane_sum(lane, minus)
            if f_plus == 0.0 or f_minus == 0.0:
                return None
            fd[axis] = (f_plus - f_minus) / (2.0 * step)
        return fd

    fd_total = torch.zeros(3)
    ad_total = torch.zeros(3)
    stable_lanes = 0
    for lane in active_indices[::5]:
        if one_lane_sum(lane, src0) <= 0.0:
            continue
        # Two-step consistency filter: a lane riding a UTD transition-region
        # kink (or an acceptance boundary that survives /-h without
        # zeroing) shows an FD that scales like 1/h instead of converging;
        # such lanes are not a valid derivative oracle and are excluded, the
        # same way the reflection fixtures avoid specular points on
        # triangulation diagonals.
        fd_coarse = lane_fd(lane, FD_STEP_POSITION)
        fd_fine = lane_fd(lane, 0.5 * FD_STEP_POSITION)
        if fd_coarse is None or fd_fine is None:
            continue
        if relative_error(fd_fine, fd_coarse, abs_floor=ABS_TOL) > 0.25:
            continue
        fd_total += fd_fine
        ad_total += lane_dual(lane)
        stable_lanes += 1
        if stable_lanes >= 48:
            break

    assert stable_lanes >= 16, "the fixture no longer yields accept-stable lanes"
    assert relative_error(ad_total, fd_total, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_diffraction_transmitter_jvp_matches_vjp():
    """Full-solve forward/reverse duality on the transmitter seed."""

    base = torch.tensor([0.0, -1.0, 0.5])
    tangent = torch.tensor([0.7, -1.1, 0.4], device="cuda")
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.clone().cuda(), tangent)
        scene = _diffraction_scene(tx=dual)
        jvp = torch.autograd.forward_ad.unpack_dual(
            _loss(_solve_diffraction(scene, "jvp"))
        ).tangent
    assert jvp is not None

    leaf = base.clone().cuda().requires_grad_(True)
    scene = _diffraction_scene(tx=leaf)
    _loss(_solve_diffraction(scene, "vjp")).backward()
    assert leaf.grad is not None
    vjp = (leaf.grad * tangent).sum()
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_diffraction_forward_mode_material_dual_matches_reverse():
    """JVP-vs-VJP inner-product duality through the diffraction map."""

    scene = _diffraction_scene()
    leaf = _material_leaf(scene, "eps_r")
    generator = torch.Generator(device="cpu").manual_seed(41)
    tangent = torch.randn(leaf.shape, generator=generator).to(leaf.device)

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(leaf.detach(), tangent)
        object.__setattr__(_compile(scene).materials, "eps_r", dual)
        try:
            jvp = torch.autograd.forward_ad.unpack_dual(
                _loss(_solve_diffraction(scene, "jvp"))
            ).tangent
        finally:
            object.__setattr__(_compile(scene).materials, "eps_r", leaf)
    assert jvp is not None

    leaf.requires_grad_(True)
    try:
        _loss(_solve_diffraction(scene, "vjp")).backward()
        vjp = (leaf.grad * tangent).sum()
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None
    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


def test_diffraction_gradient_is_stable_across_seeds():
    """Cross-seed spread of the genuinely seeded component (the AD contract).

 The diffraction map resamples its edge points and cone azimuths per
 seed, so this is the real Monte Carlo gradient-variance check; the bound
 reflects the sampling noise of the estimator at this sample count.
 """

    scene = _diffraction_scene()
    leaf = _material_leaf(scene, "eps_r")
    gradients = []
    leaf.requires_grad_(True)
    try:
        for seed in (1, 2, 3, 4, 5):
            _loss(_solve_diffraction(scene, "vjp", seed=seed)).backward()
            gradients.append(leaf.grad.detach().clone())
            leaf.grad = None
    finally:
        leaf.requires_grad_(False)
        leaf.grad = None

    stacked = torch.stack(gradients)
    # Only the wedge faces carry a diffraction gradient. Judge every
    # component's spread against the gradient's overall scale: a component
    # whose mean is a small fraction of the dominant one may legitimately
    # carry sampling noise comparable to its own mean without perturbing an
    # optimizer step, but noise at the scale of the dominant component
    # would.
    mean = stacked.mean(dim=0)
    scale = float(mean.abs().max())
    assert scale > 0.0
    spread = stacked.std(dim=0)
    assert float(spread.max()) <= 0.2 * scale
    # The dominant component itself must be seed-stable in the tight sense.
    dominant = mean.abs().argmax()
    assert float(spread[dominant] / mean[dominant].abs()) <= 0.15


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