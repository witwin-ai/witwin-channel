from __future__ import annotations

import pytest
import numpy as np
import drjit as dr
import witwin.channel as wc
import witwin.channel.path as channel_path


def test_channel_umbrella_exports_user_facing_surface():
    assert wc.Scene.__name__ == "Scene"
    assert wc.Transmitter.__name__ == "Transmitter"
    assert wc.Receiver.__name__ == "Receiver"
    assert wc.ReceiverGrid.__name__ == "ReceiverGrid"
    assert wc.Box.__name__ == "Box"
    assert wc.Material.__name__ == "Material"
    assert wc.Structure.__name__ == "Structure"
    assert wc.path.solve is not None
    assert wc.path.Tuning.__name__ == "Tuning"
    assert wc.deterministic.solve is not None
    assert wc.deterministic.Tuning.__name__ == "Tuning"
    assert wc.montecarlo.solve is not None
    assert wc.montecarlo.Tuning.__name__ == "Tuning"
    assert wc.montecarlo.IntegratorOptions.__name__ == "IntegratorOptions"


def test_channel_path_package_is_canonical_path_solver():
    assert channel_path.solve is wc.path.solve
    assert channel_path.Config is wc.path.Config
    assert channel_path.PathResult is wc.path.PathResult
    assert not hasattr(channel_path, "Result")


def test_solver_packages_do_not_export_generic_result_aliases():
    assert not hasattr(wc.path, "Result")
    assert wc.path.PathResult.__name__ == "PathResult"
    assert not hasattr(wc.deterministic, "Result")
    assert wc.deterministic.RadioMapResult.__name__ == "RadioMapResult"
    assert not hasattr(wc.montecarlo, "Result")
    assert wc.montecarlo.RadioMapResult.__name__ == "RadioMapResult"


def test_path_config_uses_interaction_neutral_names():
    config = wc.path.Config(
        num_samples=16,
        max_bounces=1,
        max_diffraction_order=0,
    )

    assert config.num_samples == 16
    assert config.max_bounces == 1
    assert config.max_diffraction_order == 0
    assert not hasattr(config, "reflection_n_rays")
    assert not hasattr(config, "reflection_max_bounces")
    assert not hasattr(config, "max_diffractions")
    assert not hasattr(config, "edge_selection_mode")
    assert not hasattr(config, "edge_diffraction")
    assert not hasattr(config, "boundary_edge_policy")
    assert not hasattr(config, "ray_mode")


def test_path_config_rejects_public_ray_mode():
    with pytest.raises(TypeError):
        wc.path.Config(ray_mode="2d")


def test_channel_umbrella_does_not_export_public_ray_mode_launcher():
    assert not hasattr(wc, "radiomap2d")


def test_core_results_own_shared_ray_mode_controls():
    from witwin.channel.core import results

    assert results.DEFAULT_RAY_MODE == "3d"
    assert results.normalize_ray_mode("2D") == "2d"


def test_path_result_does_not_export_payload_dict_constructor():
    assert not hasattr(wc.path.PathResult, "from_payload")


def test_path_solve_rejects_removed_coordinate_and_receivers_keywords():
    scene = wc.Scene(device="cpu")

    with pytest.raises(TypeError):
        wc.path.solve(scene=scene, frequency=1.0, tx_pos=(0.0, 0.0, 0.0), receiver="rx")
    with pytest.raises(TypeError):
        wc.path.solve(scene=scene, frequency=1.0, transmitter="tx", receivers=[])
    with pytest.raises(TypeError):
        wc.path.solve(scene=scene, frequency=1.0, transmitter="tx", receiver="rx", return_timing=True)


def test_path_solve_requires_scene_frequency():
    scene = wc.Scene(
        transmitters=[wc.Transmitter(name="tx", position=(0.0, 0.0, 0.0))],
        receivers=[wc.Receiver(name="rx", position=(1.0, 0.0, 0.0))],
        device="cpu",
    )

    with pytest.raises(ValueError, match="Scene.frequency"):
        wc.path.solve(scene=scene, transmitter="tx", receiver="rx")


@pytest.mark.gpu
def test_path_solve_batches_multi_tx_reflection_export(monkeypatch):
    from witwin.channel.deterministic.trace import path_export
    from witwin.channel.path import solver as path_solver

    scene = wc.Scene(
        transmitters=[
            wc.Transmitter(name="tx0", position=(0.0, 0.0, 1.0)),
            wc.Transmitter(name="tx1", position=(1.0, 0.0, 1.0)),
        ],
        receivers=[wc.Receiver(name="rx", position=(2.0, 0.0, 1.0))],
        frequency=3.5e9,
        device="cuda",
    )
    calls = []

    def fake_batched_reflection_export(**kwargs):
        calls.append(int(dr.width(kwargs["tx_positions"].x)))
        return (
            (
                path_export.empty_raw_paths(depth=1, return_geometry=kwargs["return_geometry"]),
            ),
            (None, None),
        )

    def fail_single_tx_export(**kwargs):
        raise AssertionError("multi-Tx path solve must use batched reflection export")

    monkeypatch.setattr(
        path_solver,
        "collect_reflection_paths_for_transmitters",
        fake_batched_reflection_export,
        raising=False,
    )
    monkeypatch.setattr(path_solver, "collect_reflection_paths", fail_single_tx_export)

    result = wc.path.solve(
        scene=scene,
        transmitter=["tx0", "tx1"],
        receiver="rx",
        config=wc.path.Config(num_samples=8, max_bounces=1, max_diffraction_order=0, max_num_paths=4),
    )

    assert calls == [2]
    assert np.asarray(result.num_paths).shape == (1, 1, 2, 1)


@pytest.mark.gpu
def test_path_solve_diffraction_order_uses_path_tuning_contract():
    scene = wc.Scene(
        transmitters=[wc.Transmitter(name="tx", position=(0.0, 0.0, 1.0))],
        receivers=[wc.Receiver(name="rx", position=(1.0, 0.0, 1.0))],
        frequency=3.5e9,
        device="cuda",
    )

    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(max_bounces=0, max_diffraction_order=1, max_num_paths=4),
    )

    num_paths = np.asarray(result.num_paths)
    assert int(num_paths[0, 0, 0, 0]) == 1


@pytest.mark.gpu
def test_path_solve_diffraction_with_geometry_exports_edge_object_slots():
    scene = wc.Scene(
        structures=[
            wc.Structure(
                name="wall",
                geometry=wc.Box(position=(0.0, 0.0, 1.5), size=(0.25, 4.0, 3.0), device="cuda"),
                material=wc.Material(eps_r=4.0, sigma_e=0.0),
            )
        ],
        transmitters=[wc.Transmitter(name="tx", position=(-2.0, -1.0, 1.5))],
        receivers=[wc.Receiver(name="rx", position=(-2.0, 1.0, 1.5))],
        frequency=3.5e9,
        device="cuda",
    )

    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            num_samples=64,
            max_bounces=0,
            max_diffraction_order=1,
            max_num_paths=4,
            return_geometry=True,
            edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
        ),
    )

    assert np.asarray(result.num_paths).shape == (1, 1, 1, 1)
