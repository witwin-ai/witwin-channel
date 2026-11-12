# Copyright Xingyu Chen.
# Test Monte Carlo Basic straight-penetration radiomaps.

"""Test Monte Carlo Basic straight-penetration radiomaps."""

import pytest
import torch

from tests.support.core_world import (
    make_mesh_structure,
    make_receiver_grid,
    make_transmitter,
)
from witwin.core import (
    MaterialLayer,
    PhysicalMaterial,
    ReceiverGrid,
    Scene,
    Structure,
)
from witwin.channel.deployment import build_info
from witwin.channel.radiomap import Stochastic, solve as solve_basic
from witwin.channel import radiomap
from witwin.channel.radiomap import _stochastic as mc_basic
from tests.reference.em_oracle import layer_stack_rt
from witwin.channel.runtime import (
    CapacityError,
    CapacityFailureBit,
    SolveCapacityTransaction,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA torch is required"
)

_FREQUENCY = 3.0e9


def _solve_basic(scene: Scene, config: Stochastic):
    return solve_basic(scene, config, reference_frequency_hz=_FREQUENCY)


def _require_native() -> None:
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native transmission is not built")


def _wall(material, *, x: float = 2.5) -> Structure:
    return make_mesh_structure(
        vertices=torch.tensor(
            [
                [x, -4.0, -4.0],
                [x, 4.0, -4.0],
                [x, -4.0, 4.0],
                [x, 4.0, 4.0],
            ]
        ),
        faces=torch.tensor([[0, 1, 2], [1, 3, 2]]),
        material=material,
        name=f"wall-{x}",
        surface_id=int(x * 10),
    )


def _grid(shape: tuple[int, int] = (4, 4)) -> ReceiverGrid:
    if shape == (1, 1):
        return make_receiver_grid(
            origin=torch.tensor([5.0, 0.0, 0.0]),
            x_axis=torch.tensor([0.0, 1.0, 0.0]),
            y_axis=torch.tensor([0.0, 0.0, 1.0]),
            shape=(1, 1),
            spacing=(1.0, 1.0),
        )
    return make_receiver_grid(
        origin=torch.tensor([5.0, -1.0, -0.5]),
        x_axis=torch.tensor([0.0, 1.0, 0.0]),
        y_axis=torch.tensor([0.0, 0.0, 1.0]),
        shape=shape,
        spacing=(2.0 / 3.0, 1.0 / 3.0),
    )


def _scene(structures, *, grid_shape: tuple[int, int] = (4, 4)) -> Scene:
    return Scene(
        structures=structures,
        endpoints=[
            make_transmitter(torch.tensor([0.0, 0.0, 0.0])),
            _grid(grid_shape),
        ],
    )


def _solve(scene: Scene, components, *, max_depth: int = 2):
    return _solve_basic(
        scene,
        Stochastic(samples=64, seed=3, max_depth=max_depth, components=components),
    )


def test_vacuum_wall_transmission_map_equals_unobstructed_los_map():
    _require_native()
    vacuum = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.2, eps_r=1.0),),
        name="vacuum-wall",
    )
    walled = _solve(_scene([_wall(vacuum)]), {"los", "transmission"})
    empty = _solve(_scene([]), {"los"})

    # The wall blocks every tx->cell segment: the exclusive los class is zero.
    assert torch.count_nonzero(walled.component_maps["los"]) == 0
    # A vacuum layer has unit power transmittance, so the transmission map
    # reproduces the unobstructed analytic LoS map (acceptance test).
    torch.testing.assert_close(
        walled.component_maps["transmission"],
        empty.component_maps["los"],
        rtol=1.0e-4,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        walled.component_power["transmission"],
        walled.component_maps["transmission"].sum(),
        rtol=1.0e-5,
        atol=1.0e-10,
    )
    assert walled.metadata["components"]["transmission"] == "enabled"


def test_lossy_wall_attenuates_by_stack_power_transmittance():
    _require_native()
    thickness, eps_r, sigma_e = 0.1, 4.0, 0.05
    lossy = PhysicalMaterial(
        layers=(
            MaterialLayer(
                thickness_m=thickness,
                eps_r=eps_r,
                sigma_e=sigma_e,
            ),
        ),
        name="lossy-wall",
    )
    # 1x1 grid straight behind the wall: exact normal incidence.
    walled = _solve(_scene([_wall(lossy)], grid_shape=(1, 1)), {"transmission"})
    empty = _solve(_scene([], grid_shape=(1, 1)), {"los"})

    oracle = layer_stack_rt([(thickness, eps_r, sigma_e, 1.0)], 1.0, _FREQUENCY)
    # transmission polarization: the per-wall transmittance is the Jones-derived power projected on
    # the incident polarization, not the unpolarized TE/TM mean. At this exact
    # normal incidence the plane of incidence is degenerate and T_te == T_tm, so
    # the polarized projection is simply T_te.
    expected_t = float(oracle.T_te)
    assert float(oracle.T_te) == pytest.approx(float(oracle.T_tm), rel=1.0e-6)
    assert 0.0 < expected_t < 1.0
    torch.testing.assert_close(
        walled.component_maps["transmission"],
        empty.component_maps["los"] * expected_t,
        rtol=5.0e-4,
        atol=1.0e-14,
    )


def test_pec_wall_transmits_nothing():
    _require_native()
    walled = _solve(
        _scene([_wall(PhysicalMaterial.perfect_conductor())]),
        {"transmission"},
    )
    assert float(walled.component_maps["transmission"].abs().max()) < 1.0e-20
    assert float(walled.component_power["transmission"]) < 1.0e-20
    assert walled.metadata["contribution_capacity"] == 16


def test_transmission_exact_capacity_recovers_unobstructed_map():
    _require_native()
    vacuum = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.2, eps_r=1.0),),
        name="vacuum-wall",
    )
    scene = _scene([_wall(vacuum, x=2.0), _wall(vacuum, x=3.0)])
    empty = _solve(_scene([]), {"los"})

    # Exactly D=2 accepted hits succeeds; the mandatory D+1 probe sees a
    # clear tail and does not report overflow.
    full = _solve(scene, {"transmission"}, max_depth=2)
    torch.testing.assert_close(
        full.component_maps["transmission"],
        empty.component_maps["los"],
        rtol=1.0e-4,
        atol=1.0e-12,
    )


def test_transmission_over_depth_cells_are_dropped_and_counted():
    _require_native()
    vacuum = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.2, eps_r=1.0),),
        name="vacuum-wall",
    )
    scene = _scene([_wall(vacuum, x=2.0), _wall(vacuum, x=3.0)])
    result = _solve(
        scene, {"los", "reflection", "diffraction", "transmission"}, max_depth=1
    )
    torch.cuda.synchronize()

    # Every grid cell sits behind both walls: two crossings exceed max_depth=1,
    # so each cell's transmission row is dropped and counted, not failed.
    cells = int(result.component_maps["transmission"].numel())
    assert result.metadata["transmission"]["dropped_rows_over_max_depth"] == cells
    assert torch.count_nonzero(result.component_maps["transmission"]).item() == 0
    assert torch.isfinite(result.path_gain).all().item()
    # The same scene at max_depth=2 keeps every cell.
    deeper = _solve(scene, {"transmission"}, max_depth=2)
    assert deeper.metadata["transmission"]["dropped_rows_over_max_depth"] == 0
    assert torch.count_nonzero(deeper.component_maps["transmission"]).item() == cells


def test_transmission_failure_bit_raises_capacity_error_without_a_result(
    monkeypatch: pytest.MonkeyPatch,
):
    _require_native()
    original = radiomap._stochastic.straight_transmission_chains

    def failing(*args, failure_state, **kwargs):
        penetration = original(*args, failure_state=failure_state, **kwargs)
        failure_state.bits.fill_(int(CapacityFailureBit.SEGMENT_PENETRATION_FAILURE))
        return penetration

    monkeypatch.setattr(mc_basic, "straight_transmission_chains", failing)
    vacuum = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.2, eps_r=1.0),),
        name="vacuum-wall",
    )
    with pytest.raises(CapacityError) as caught:
        _solve(_scene([_wall(vacuum)]), {"los", "transmission"}, max_depth=1)
    assert caught.value.bit_names == ("SEGMENT_PENETRATION_FAILURE",)
    assert float(torch.ones(2, device="cuda").sum()) == 2.0


def test_transmission_solve_shares_exact_failure_state_through_final_terminal(
    monkeypatch: pytest.MonkeyPatch,
):
    _require_native()
    original_component = radiomap._stochastic.transmission_component_map
    original_sanitize = radiomap._stochastic.mc_capacity_failure_component_maps_sanitize
    observed = {}

    def component(*args, failure_state, **kwargs):
        observed["component"] = failure_state
        return original_component(*args, failure_state=failure_state, **kwargs)

    def sanitize(*args, failure_state, **kwargs):
        observed["sanitize"] = failure_state
        return original_sanitize(*args, failure_state=failure_state, **kwargs)

    def terminal(transaction: SolveCapacityTransaction) -> int:
        observed["terminal"] = transaction.failure_state
        return 0

    monkeypatch.setattr(mc_basic, "transmission_component_map", component)
    monkeypatch.setattr(
        mc_basic,
        "mc_capacity_failure_component_maps_sanitize",
        sanitize,
    )
    monkeypatch.setattr(SolveCapacityTransaction, "terminal_check", terminal)
    vacuum = PhysicalMaterial(
        layers=(MaterialLayer(thickness_m=0.2, eps_r=1.0),),
        name="vacuum-wall",
    )
    result = _solve(
        _scene([_wall(vacuum)], grid_shape=(1, 1)),
        {"transmission"},
        max_depth=1,
    )
    torch.cuda.synchronize()

    assert set(observed) == {"component", "sanitize", "terminal"}
    assert observed["component"] is observed["sanitize"] is observed["terminal"]
    assert (
        observed["component"].bits.data_ptr()
        == observed["sanitize"].bits.data_ptr()
        == observed["terminal"].bits.data_ptr()
    )
    assert observed["component"].bits.tolist() == [0]
    assert result.component_maps is not None
    assert torch.count_nonzero(result.component_maps["transmission"]).item() == 1