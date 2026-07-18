"""ADR-015 Part A: montecarlo.basic Kirchhoff scattering-map gradients.

The MC-basic scattering radiomap is an area-sampling estimator whose only
native physics op in the contribution weight is the resident Kirchhoff table
lookup ``scattering_table_eval`` (ADR-015 op 1). This file pins the three
gradient chains ADR-015 Part A adds and the frozen-winner / no-graph contracts:

- BSDF table values (``f_te``/``f_tm``) through the native table companions;
- transmitter power, a linear scale on the whole map;
- frequency, which Part A carries ONLY through the ``(lambda/4pi)^2``
  amplitude factor. The resident table is a compile-time float64 island built
  at a detached frequency (ScatteringResourceKey does not depend on frequency,
  so a compiled scene reuses one table), and the ADR-014/015 Part C
  table-frequency chain is a separate change. The frequency finite difference
  here therefore reuses ONE compiled scene (table pinned) and moves only the
  amplitude carrier, matching the Part A partial derivative. A full-scene
  frequency FD that rebuilt the table would also measure the table's frequency
  dependence and is out of scope until Part C lands. DEVIATION flagged.

The area-sample set, both binary visibility masks (tx->point, point->rx) and
the incidence gate stay frozen winners in AD (plan-07 fixed-topology contract);
the map is exactly linear in the table values and in tx power, so those two
finite differences are analytic up to float noise. FD parity uses the shared
``tests/ad/_fd`` engine on the same seeded topology as the AD pass.
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
    REL_TOL_GENERAL,
)
from witwin.channel_native import (
    ReceiverGrid,
    Scene,
    Structure,
    Transmitter,
)
from witwin.channel_native.core.kernels.extension import build_info
from witwin.channel_native.core.materials import Layer, PhysicalSurface, Roughness
from witwin.channel_native.montecarlo.basic import Config, solve
from witwin.channel_native.montecarlo.events.scattering import (
    rough_material_runtimes,
    scattering_map_matrix,
)
from witwin.channel_native.scene.tensors import (
    receiver_grid_points,
    transmitter_positions,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for solver AD"
)

_FREQUENCY_HZ = 60.0e9
_SAMPLES = 4096
_SEED = 7
# Strongly diffuse Kirchhoff lobe inside the applicability domain (k0*l ~ 12.6),
# so the area estimator deposits nonzero power at the receiver grid.
_SIGMA_H = 1.0e-3
_CORR = 0.01
_EPS_R = 4.0
_SIGMA_E = 0.05
_THICKNESS = 0.1


def _require_native() -> None:
    if not build_info()["uses_raydn_native"]:
        pytest.skip("RayDN native scattering is not built")


def _wall() -> Structure:
    return Structure(
        vertices=torch.tensor(
            [[2.5, -4.0, -4.0], [2.5, 4.0, -4.0], [2.5, -4.0, 4.0], [2.5, 4.0, 4.0]]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=PhysicalSurface(
            layers=(Layer(thickness_m=_THICKNESS, eps_r=_EPS_R, sigma_e=_SIGMA_E),),
            roughness_front=Roughness(
                rms_height_m=_SIGMA_H, corr_length_x_m=_CORR, corr_length_y_m=_CORR
            ),
            name="rough-wall",
        ),
        name="wall",
        surface_id=1,
    )


def _grid() -> ReceiverGrid:
    # Receiver grid on the transmitter side of the wall, near the specular lobe.
    return ReceiverGrid(
        origin=torch.tensor([0.5, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(4, 4),
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _scene(frequency: float | torch.Tensor = _FREQUENCY_HZ) -> Scene:
    return Scene(
        structures=[_wall()],
        transmitters=[Transmitter(position=torch.tensor([0.0, 0.0, 0.0]))],
        receivers=[_grid()],
        frequency=frequency,
    )


def _map_handles(scene: Scene):
    device = torch.device("cuda")
    raydn = scene.raydn_scene()
    tx_pos, tx_power = transmitter_positions(scene, device=device)
    grid = scene.receivers[0]
    rx_pos = receiver_grid_points(grid, reference=tx_pos)
    return device, raydn, tx_pos, tx_power, rx_pos


def _matrix(scene, raydn, tx_pos, tx_power, rx_pos, *, ad: bool):
    matrix, stats = scattering_map_matrix(
        scene,
        raydn,
        tx_pos,
        tx_power,
        rx_pos,
        samples=_SAMPLES,
        seed=_SEED,
        device=tx_pos.device,
        ad=ad,
    )
    return matrix, stats


# ---------------------------------------------------------------------------
# BSDF table-value gradient (the native op-1 chain).
# ---------------------------------------------------------------------------


def test_scattering_table_value_grad_matches_fd():
    _require_native()
    scene = _scene()
    device, raydn, tx_pos, tx_power, rx_pos = _map_handles(scene)
    runtimes = rough_material_runtimes(scene.compile())
    assert runtimes, "the fixture must have a rough material"
    table = next(iter(runtimes.values())).table
    f_te = table.f_te

    # Sanity: the map deposits nonzero power on this fixture.
    base_matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=False)
    assert float(base_matrix.abs().sum()) > 0.0

    base = f_te.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(17)
    tangent = torch.randn(base.shape, generator=generator).to(base)

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f_te.copy_(value.to(device=f_te.device, dtype=f_te.dtype))
        try:
            matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=False)
            return matrix.sum().detach()
        finally:
            with torch.no_grad():
                f_te.copy_(base)

    expected = central_difference_directional(evaluate, base, tangent, 1.0e-2)

    f_te.requires_grad_(True)
    try:
        matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=True)
        assert matrix.requires_grad
        matrix.sum().backward()
        assert f_te.grad is not None
        got = (f_te.grad.detach() * tangent).sum()
    finally:
        f_te.requires_grad_(False)
        f_te.grad = None
        with torch.no_grad():
            f_te.copy_(base)

    assert relative_error(got, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# Transmitter-power gradient (a linear scale on the whole map).
# ---------------------------------------------------------------------------


def test_scattering_tx_power_grad_matches_fd():
    _require_native()
    scene = _scene()
    device, raydn, tx_pos, tx_power_native, rx_pos = _map_handles(scene)

    base = tx_power_native.detach().clone()

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        matrix, _ = _matrix(
            scene, raydn, tx_pos, value.to(base), rx_pos, ad=False
        )
        return matrix.sum().detach()

    expected = central_difference_gradient(evaluate, base, 1.0e-3)

    tx_power = base.clone().requires_grad_(True)
    matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=True)
    matrix.sum().backward()
    assert tx_power.grad is not None
    assert float(tx_power.grad.abs().max()) > 0.0
    assert relative_error(tx_power.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# Frequency gradient (the amplitude^2 carrier only; table pinned, Part A).
# ---------------------------------------------------------------------------


def test_scattering_frequency_grad_matches_fd():
    _require_native()
    frequency = torch.tensor(_FREQUENCY_HZ, device="cuda").requires_grad_(True)
    scene = _scene(frequency=frequency)
    # Compile once so the Kirchhoff table is pinned at the base frequency; the
    # FD below moves only the amplitude carrier, matching Part A.
    device, raydn, tx_pos, tx_power, rx_pos = _map_handles(scene)
    assert rough_material_runtimes(scene.compile())

    base = frequency.detach().clone()

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            frequency.copy_(value.to(device=frequency.device, dtype=frequency.dtype))
        try:
            matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=False)
            return matrix.sum().detach()
        finally:
            with torch.no_grad():
                frequency.copy_(base)

    step = FD_REL_STEP_FREQUENCY * _FREQUENCY_HZ
    expected = central_difference_gradient(evaluate, base, step)

    matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=True)
    assert matrix.requires_grad
    matrix.sum().backward()
    assert frequency.grad is not None
    assert relative_error(frequency.grad, expected, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# JVP-vs-VJP duality through the map (table-value tangent).
# ---------------------------------------------------------------------------


def test_scattering_forward_mode_table_dual_matches_reverse():
    _require_native()
    scene = _scene()
    device, raydn, tx_pos, tx_power, rx_pos = _map_handles(scene)
    table = next(iter(rough_material_runtimes(scene.compile()).values())).table
    f_te = table.f_te
    base = f_te.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(23)
    tangent = torch.randn(base.shape, generator=generator).to(base)

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base.clone(), tangent)
        object.__setattr__(table, "f_te", dual)
        try:
            matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=True)
            jvp = torch.autograd.forward_ad.unpack_dual(matrix.sum()).tangent
        finally:
            # Restore the ORIGINAL leaf tensor object (not a clone) so the
            # reverse pass below sees the same graph input.
            object.__setattr__(table, "f_te", f_te)
    assert jvp is not None

    f_te.requires_grad_(True)
    try:
        matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=True)
        matrix.sum().backward()
        vjp = (f_te.grad.detach() * tangent).sum()
    finally:
        f_te.requires_grad_(False)
        f_te.grad = None

    assert relative_error(jvp, vjp, abs_floor=ABS_TOL) <= REL_TOL_GENERAL


# ---------------------------------------------------------------------------
# Frozen topology + primal-preservation.
# ---------------------------------------------------------------------------


def test_scattering_ad_preserves_frozen_primal_map():
    """Same seed -> identical map; ad on -> identical primal (frozen winners)."""

    _require_native()
    scene = _scene()
    device, raydn, tx_pos, tx_power, rx_pos = _map_handles(scene)

    a, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=False)
    b, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=False)
    # The area-sample set and both visibility masks are seed-deterministic.
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)

    # The AD path runs the same native forward and torch arithmetic on the
    # same frozen sample set, so the primal value is unchanged (no live leaf
    # here, so the AD matrix carries no graph and that is fine).
    ad_matrix, _ = _matrix(scene, raydn, tx_pos, tx_power, rx_pos, ad=True)
    torch.testing.assert_close(ad_matrix.detach(), a, rtol=1.0e-6, atol=0.0)


# ---------------------------------------------------------------------------
# End-to-end solver: ad_mode="none" no-graph, vjp reaches the table leaf.
# ---------------------------------------------------------------------------


def _solve(scene: Scene, ad_mode: str):
    return solve(
        scene,
        Config(
            samples=_SAMPLES,
            seed=_SEED,
            components={"scattering"},
            max_depth=1,
            ad_mode=ad_mode,
        ),
    )


def test_scattering_solver_ad_mode_none_builds_no_graph():
    _require_native()
    scene = _scene()
    result = _solve(scene, "none")
    assert not result.path_gain.requires_grad
    assert result.path_gain.grad_fn is None
    assert result.metadata["kernel"]["ad_status"] == "none"


def test_scattering_solver_vjp_reaches_table_leaf():
    _require_native()
    scene = _scene()
    table = next(iter(rough_material_runtimes(scene.compile()).values())).table
    f_te = table.f_te
    f_te.requires_grad_(True)
    try:
        result = _solve(scene, "vjp")
        assert result.path_gain.requires_grad
        assert result.metadata["kernel"]["ad_status"] == "vjp"
        result.path_gain.sum().backward()
        assert f_te.grad is not None
        assert float(f_te.grad.abs().sum()) > 0.0
    finally:
        f_te.requires_grad_(False)
        f_te.grad = None
