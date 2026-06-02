from dataclasses import replace

import pytest
import torch
import witwin.channel as wt
import witwin.channel as wc
import witwin.channel.deterministic as deterministic
import witwin.channel.montecarlo as montecarlo

from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
from witwin.channel.core.results import (
    RadioMapCoordinates,
    RadioMapFieldPayload,
    RadioMapPowerPayload,
    RadioMapResult,
)
from witwin.channel.core.numerics.tensors import to_torch_view, to_float_tensor
from witwin.core import Box, Material, Structure


pytestmark = pytest.mark.gpu


def _contract_scene() -> Scene:
    return Scene(
        structures=[
            Structure(
                name="distant_block",
                geometry=Box(
                    position=(5.0, 5.0, 5.0),
                    size=(0.25, 0.25, 0.25),
                    device="cuda",
                ),
                material=Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        transmitters=[
            Transmitter(
                name="tx",
                position=wt.Point3f(0.0, 0.0, 1.0),
                power=2.5,
            ),
            Transmitter(
                name="tx2",
                position=wt.Point3f(0.5, 0.0, 1.0),
                power=1.5,
            )
        ],
        receivers=[
            ReceiverGrid(
                name="rm",
                axis="z",
                position=1.0,
                bounds=((-1.0, 1.0), (-1.0, 1.0)),
                grid_shape=(3, 2),
            )
        ],
        frequency=1.0e9,
        device="cuda",
    )


def test_shared_radiomap_result_types_are_importable():
    assert RadioMapCoordinates.__name__ == "RadioMapCoordinates"
    assert RadioMapResult.__name__ == "RadioMapResult"
    assert RadioMapFieldPayload.__name__ == "RadioMapFieldPayload"
    assert RadioMapPowerPayload.__name__ == "RadioMapPowerPayload"
    assert deterministic.RadioMapResult is RadioMapResult
    assert montecarlo.RadioMapResult is RadioMapResult
    assert not hasattr(deterministic, "Result")
    assert not hasattr(montecarlo, "Result")
    assert not hasattr(deterministic, "GridSpec")
    assert not hasattr(montecarlo, "GridSpec")


def test_deterministic_result_uses_shared_contract():
    det = deterministic.solve(
        scene=_contract_scene(),
        transmitter="tx",
        receiver="rm",
        config=deterministic.Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            shadow_boundary_correction=False,
        ),
    )

    assert det.solver == "deterministic"
    assert type(det) is RadioMapResult
    assert det.field is not None
    assert det.power is None
    assert det.coords.sample_positions
    assert det.path_gain.shape == det.rss.shape == det.sinr.shape
    assert tuple(det.path_gain.shape) == (1, 2, 3)
    assert tuple(det.best_tx_index.shape) == (2, 3)


def test_monte_carlo_result_uses_shared_contract():
    mc = montecarlo.solve(
        scene=_contract_scene(),
        transmitter="tx",
        receiver="rm",
        config=montecarlo.Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            tuning=montecarlo.Tuning(shadow_boundary_mode="none"),
            integrator_options=montecarlo.IntegratorOptions(
                integrator="basic",
                samples_per_tx=8,
                ad=False,
            ),
        ),
    )

    assert mc.solver == "montecarlo"
    assert type(mc) is RadioMapResult
    assert mc.field is None
    assert mc.power is not None
    assert mc.power.incoherent is mc.incoherent
    assert mc.coords.sample_positions == ()
    assert mc.path_gain.shape == mc.rss.shape == mc.sinr.shape
    assert tuple(mc.path_gain.shape) == (1, 2, 3)
    assert tuple(mc.best_tx_index.shape) == (2, 3)
    assert "timing" in mc.metadata
    assert mc.timing is not None


def test_scene_receiver_grid_contract_matches_across_solvers():
    scene = _contract_scene()
    det = deterministic.solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=deterministic.Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            shadow_boundary_correction=False,
        ),
    )
    mc = montecarlo.solve(
        scene=scene,
        transmitter="tx",
        receiver="rm",
        config=montecarlo.Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            tuning=montecarlo.Tuning(shadow_boundary_mode="none"),
            integrator_options=montecarlo.IntegratorOptions(
                integrator="basic",
                samples_per_tx=8,
                ad=False,
            ),
        ),
    )

    assert det.grid_shape == mc.grid_shape == (3, 2)
    assert det.cell_size == mc.cell_size
    assert det.surface["axis"] == mc.surface["axis"] == "z"
    assert det.surface["bounds"] == mc.surface["bounds"]
    assert det.coords.grid_x.shape == mc.coords.grid_x.shape
    assert det.coords.grid_y.shape == mc.coords.grid_y.shape
    assert det.coords.cell_centers.shape == mc.coords.cell_centers.shape


def test_radiomap_solvers_reject_removed_coordinate_and_grid_keywords():
    scene = _contract_scene()
    with pytest.raises(TypeError):
        deterministic.solve(
            scene=scene,
            frequency=1.0e9,
            tx_pos=wt.Point3f(0.0, 0.0, 1.0),
            receiver="rm",
        )
    with pytest.raises(TypeError):
        montecarlo.solve(
            scene=scene,
            frequency=1.0e9,
            transmitter="tx",
            grid=scene.receiver("rm"),
        )
    with pytest.raises(TypeError):
        montecarlo.solve(
            scene=scene,
            frequency=1.0e9,
            transmitter="tx",
            receiver="rm",
            return_timing=True,
        )


def test_radiomap_config_rejects_old_field_names():
    with pytest.raises(TypeError):
        deterministic.Config(reflection_n_rays=8)
    with pytest.raises(TypeError):
        deterministic.Config(reflection_max_bounces=0)
    with pytest.raises(TypeError):
        deterministic.Config(max_diffractions=0)
    with pytest.raises(TypeError):
        deterministic.Config(edge_selection_mode="all_edges")
    with pytest.raises(TypeError):
        deterministic.Config(ray_mode="2d")
    with pytest.raises(TypeError):
        deterministic.Config(enable_rd_diffraction=True)
    with pytest.raises(TypeError):
        deterministic.Config(solver_mode="memory_safe")
    with pytest.raises(TypeError):
        montecarlo.Config(reflection_n_rays=8)
    with pytest.raises(TypeError):
        montecarlo.Config(reflection_max_bounces=0)
    with pytest.raises(TypeError):
        montecarlo.Config(max_diffractions=0)
    with pytest.raises(TypeError):
        montecarlo.Config(edge_selection_mode="all_edges")
    with pytest.raises(TypeError):
        montecarlo.Config(integrator="bdpt")
    with pytest.raises(TypeError):
        montecarlo.Config(shadow_boundary_backend="drjit")


def test_radiomap_multi_tx_contract_and_helpers():
    result = deterministic.solve(
        scene=_contract_scene(),
        transmitter=["tx", "tx2"],
        receiver="rm",
        config=deterministic.Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            shadow_boundary_correction=False,
        ),
    )

    assert tuple(result.path_gain.shape) == (2, 2, 3)
    assert tuple(result.rss.shape) == (2, 2, 3)
    assert tuple(result.sinr.shape) == (2, 2, 3)
    assert tuple(result.best_tx_index.shape) == (2, 3)
    assert tuple(result.cell_association().shape) == (2, 3)
    rss = to_torch_view(result.rss, dtype=torch.float32)
    best = to_torch_view(result.cell_association(), dtype=torch.int32)
    assert torch.equal(best, torch.argmax(rss, dim=0).to(dtype=torch.int32))

    single = result.squeeze_tx(0)
    assert tuple(single.rss.shape) == (2, 3)


def test_radiomap_squeeze_tx_resets_single_tx_association_map():
    result = deterministic.solve(
        scene=_contract_scene(),
        transmitter=["tx", "tx2"],
        receiver="rm",
        config=deterministic.Config(
            num_samples=8,
            max_bounces=0,
            max_diffraction_order=0,
            shadow_boundary_correction=False,
        ),
    )
    crafted_rss = torch.tensor(
        [
            [[1.0, 4.0, 1.0], [1.0, 1.0, 3.0]],
            [[2.0, 1.0, 5.0], [4.0, 2.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    crafted = replace(
        result,
        path_gain=to_float_tensor(crafted_rss, shape=(2, 2, 3)),
        rss=to_float_tensor(crafted_rss, shape=(2, 2, 3)),
        sinr=to_float_tensor(crafted_rss, shape=(2, 2, 3)),
    )

    squeezed = crafted.squeeze_tx(0)
    best = to_torch_view(squeezed.best_tx_index, dtype=torch.int32)
    association = to_torch_view(squeezed.cell_association(), dtype=torch.int32)

    assert tuple(squeezed.rss.shape) == (2, 3)
    assert torch.count_nonzero(best).item() == 0
    assert torch.count_nonzero(association).item() == 0
