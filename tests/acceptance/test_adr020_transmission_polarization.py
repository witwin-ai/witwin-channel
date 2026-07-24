"""ADR-020 transmission polarization unification parity.

The transmission MODEL for the montecarlo.basic and montecarlo.bdpt solvers is
the same full-Jones layer-stack evaluation the deterministic/Path solvers use.
On a polarized oblique-incidence wall the two Monte Carlo solvers must therefore
reproduce the deterministic polarized transmittance (each in its own estimator
domain), not the polarization-agnostic TE/TM mean.

- BDPT routes pure transmission through the shared enumerated engine, so its
  transmission component reproduces the deterministic value exactly (same native
  full-Jones field, receiver-antenna projection included).
- MC basic keeps its power-domain radiomap: its per-wall transmittance is the
  Jones-derived power projected on the incident polarization (no receiver
  projection), which equals f_te*T_te + f_tm*T_tm, not the mean.
"""

import math

import pytest
import torch

from tests.support.core_world import make_receiver_grid, make_transmitter
from tests.support.scenes import transmission_wall_structure
from witwin.channel.deployment import build_info
from tests.reference.em_oracle import layer_stack_rt
from witwin.core import MaterialLayer, PhysicalMaterial, ReceiverGrid, Scene

from witwin.channel.deterministic import Config as DetConfig, solve as det_solve
from witwin.channel.montecarlo.basic import Config as McConfig, solve as mc_solve
from witwin.channel.montecarlo.bdpt import Config as BdptConfig, solve as bdpt_solve

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)

_FREQUENCY = 3.0e9
_WALL_X = 2.5
# 45 degree oblique incidence on the x=_WALL_X wall (normal +x): the tx->cell
# segment from the origin to [5,5,0] makes cos(theta)=1/sqrt(2) with the normal.
_RX = [5.0, 5.0, 0.0]
_COS_THETA = math.cos(math.pi / 4.0)
# Lossy dielectric so TE and TM transmittances differ appreciably at oblique
# incidence (mean != polarized projection).
_LAYER = (0.1, 4.0, 0.05, 1.0)


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native transmission is not built")


def _wall() -> object:
    return transmission_wall_structure(
        _WALL_X,
        PhysicalMaterial(
            layers=(
                MaterialLayer(
                    thickness_m=_LAYER[0],
                    eps_r=_LAYER[1],
                    sigma_e=_LAYER[2],
                ),
            ),
            name="adr020-lossy",
        ),
    )


def _grid() -> ReceiverGrid:
    # Single cell centered at _RX -> exact 45 degree oblique incidence.
    return make_receiver_grid(
        origin=torch.tensor(_RX),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=(1, 1),
        spacing=(1.0, 1.0),
    )


def _scene(structures, *, polarization) -> Scene:
    return Scene(
        structures=structures,
        endpoints=[
            make_transmitter(
                position=torch.tensor([0.0, 0.0, 0.0]),
                polarization=torch.tensor(polarization),
            ),
            _grid(),
        ],
    )


def _t(result) -> float:
    return float(result.component_power["transmission"].sum())


def _los(result) -> float:
    return float(result.component_power["los"].sum())


def _budgets() -> tuple[float, float]:
    oracle = layer_stack_rt([_LAYER], _COS_THETA, _FREQUENCY)
    return float(oracle.T_te), float(oracle.T_tm)


def test_pure_te_oblique_all_solvers_match_polarized_te_not_mean():
    """Default z polarization is pure TE at this geometry: all three solvers
    reproduce the deterministic polarized transmittance T_te, distinctly below
    the retired unpolarized mean 0.5*(T_te+T_tm)."""

    _require_native()
    t_te, t_tm = _budgets()
    mean = 0.5 * (t_te + t_tm)
    # The fixture must actually separate the polarized value from the mean.
    assert abs(t_tm - t_te) / t_te > 0.2

    pol = [0.0, 0.0, 1.0]  # pure TE (z) at this geometry
    det = det_solve(
        _scene([_wall()], polarization=pol),
        DetConfig(components={"transmission"}, max_depth=1),
        reference_frequency_hz=_FREQUENCY,
    )
    det_los = det_solve(
        _scene([], polarization=pol),
        DetConfig(components={"los"}),
        reference_frequency_hz=_FREQUENCY,
    )
    mc = mc_solve(
        _scene([_wall()], polarization=pol),
        McConfig(samples=64, seed=3, max_depth=1, components={"transmission"}),
        reference_frequency_hz=_FREQUENCY,
    )
    mc_los = mc_solve(
        _scene([], polarization=pol),
        McConfig(samples=64, seed=3, components={"los"}),
        reference_frequency_hz=_FREQUENCY,
    )
    bd = bdpt_solve(
        _scene([_wall()], polarization=pol),
        BdptConfig(samples=1024, seed=5, max_depth=1, components={"transmission"}),
        reference_frequency_hz=_FREQUENCY,
    )

    # Intra-solver transmission/LoS ratio isolates the transmittance model from
    # each solver's LoS convention. Every solver lands on the polarized T_te.
    assert _t(det) / _los(det_los) == pytest.approx(t_te, rel=2.0e-3)
    assert _t(mc) / _los(mc_los) == pytest.approx(t_te, rel=2.0e-3)
    # BDPT reproduces the deterministic component power exactly (shared native
    # full-Jones enumerated field), not merely the ratio.
    assert _t(bd) / _t(det) == pytest.approx(1.0, rel=2.0e-3)
    # Regression guard: the retired mean would have given a clearly larger ratio.
    assert _t(mc) / _los(mc_los) < mean * (1.0 - 0.02)


def test_mixed_polarization_bdpt_matches_deterministic_mc_is_incident_projected():
    """A mixed TE/TM polarization exposes the per-solver estimator domain: BDPT
    matches the deterministic receiver-projected value, MC basic yields the
    incident-projected power-domain transmittance f_te*T_te + f_tm*T_tm."""

    _require_native()
    t_te, t_tm = _budgets()
    mean = 0.5 * (t_te + t_tm)
    # POL is fully transverse to the tx->cell direction [1,1,0]/sqrt(2) with
    # TE(z) fraction 2/3 and TM([1,-1,0]) fraction 1/3 at this geometry.
    pol = [1.0, -1.0, 2.0]
    incident_projected = (2.0 / 3.0) * t_te + (1.0 / 3.0) * t_tm

    det = det_solve(
        _scene([_wall()], polarization=pol),
        DetConfig(components={"transmission"}, max_depth=1),
        reference_frequency_hz=_FREQUENCY,
    )
    mc = mc_solve(
        _scene([_wall()], polarization=pol),
        McConfig(samples=64, seed=3, max_depth=1, components={"transmission"}),
        reference_frequency_hz=_FREQUENCY,
    )
    mc_los = mc_solve(
        _scene([], polarization=pol),
        McConfig(samples=64, seed=3, components={"los"}),
        reference_frequency_hz=_FREQUENCY,
    )
    bd = bdpt_solve(
        _scene([_wall()], polarization=pol),
        BdptConfig(samples=1024, seed=5, max_depth=1, components={"transmission"}),
        reference_frequency_hz=_FREQUENCY,
    )

    # BDPT reproduces the deterministic component power (both receiver-projected
    # full-Jones enumerated fields).
    assert _t(bd) / _t(det) == pytest.approx(1.0, rel=2.0e-3)
    # MC basic radiomap carries the incident-projected power-domain transmittance
    # (no receiver projection), distinct from both the deterministic value and
    # the retired mean.
    assert _t(mc) / _los(mc_los) == pytest.approx(incident_projected, rel=2.0e-3)
    assert abs(incident_projected - mean) / mean > 0.02
    # Acceptance gate: both MC solvers stay within [0.5x, 2x] of deterministic.
    assert 0.5 <= _t(mc) / _t(det) <= 2.0
    assert 0.5 <= _t(bd) / _t(det) <= 2.0
