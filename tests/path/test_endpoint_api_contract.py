from __future__ import annotations

from types import SimpleNamespace

import pytest
import numpy as np
import drjit as dr
import witwin.channel as wc
import witwin.channel.path as channel_path
from witwin.channel import types as wt
from witwin.channel.deterministic.types import InteractionType


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


def test_diffraction_cached_no_geometry_assembly_skips_full_state_replay(monkeypatch):
    from witwin.channel.deterministic.trace import path_export_assembly

    def fail_materialize(*args, **kwargs):
        raise AssertionError("cached no-geometry diffraction paths should not replay full states")

    def fake_type_slots(raw, path_indices=None):
        return (
            dr.full(
                wt.Int32,
                InteractionType.DIFFRACTION,
                0 if path_indices is None else int(dr.width(path_indices)),
            ),
        )

    monkeypatch.setattr(
        path_export_assembly,
        "_materialize_diffraction_state_path_refs",
        fail_materialize,
    )
    monkeypatch.setattr(
        path_export_assembly,
        "_diffraction_state_ref_type_slots",
        fake_type_slots,
    )
    rx_positions = wt.Point3f(wt.Float([1.0]), wt.Float([0.0]), wt.Float([0.0]))
    tx_positions = wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0]))
    raw = {
        "payload_kind": "diffraction_state_refs_v1",
        "rx_index": wt.UInt32([0]),
        "tx_index": wt.UInt32([0]),
        "local_rx_index": wt.UInt32([0]),
        "state_idx": wt.UInt32([0]),
        "a": wt.Complex2f(wt.Float([1.0]), wt.Float([0.0])),
        "tau": wt.Float([1.0]),
        "theta_t": wt.Float([0.25]),
        "phi_t": wt.Float([0.5]),
        "theta_r": wt.Float([0.75]),
        "phi_r": wt.Float([1.0]),
        "path_depth": wt.UInt32([1]),
        "tx_pos": wt.Point3f(0.0, 0.0, 0.0),
        "rx_positions": rx_positions,
        "state_arrays": {"n_states": 1},
        "edge_data": None,
        "edge_object_idx": None,
        "metadata": {"n_paths": 1},
    }

    payload = path_export_assembly.assemble_result_payload(
        name="rx",
        num_tx=1,
        num_rx=1,
        max_num_paths=1,
        tx_pos=(0.0, 0.0, 0.0),
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        frequency=3.5e9,
        wavelength=0.1,
        raw_collections=[raw],
        return_geometry=False,
    )

    assert bool(payload["valid"][0, 0, 0])
    assert int(payload["types"][0, 0, 0, 0]) == InteractionType.DIFFRACTION


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
@pytest.mark.parametrize(
    ("backend", "expected_prefer", "expected_require", "expected_metadata_backend"),
    (
        (None, True, True, "rayd_optix"),
        ("native", True, True, "rayd_optix"),
        ("drjit", False, False, "drjit"),
    ),
)
def test_path_solve_reflection_backend_controls_rayd_epc(
    monkeypatch,
    backend,
    expected_prefer,
    expected_require,
    expected_metadata_backend,
):
    from witwin.channel.deterministic.trace import path_export

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
    epc_calls = []

    def fake_epc(**kwargs):
        epc_calls.append(
            (
                kwargs.get("prefer_rayd_epc"),
                kwargs.get("require_rayd_epc"),
            )
        )
        width = int(dr.width(kwargs["target_pos"].x))
        zero = wt.Complex2f(dr.zeros(wt.Float, width), dr.zeros(wt.Float, width))
        valid = dr.full(wt.Bool, False, width)
        point = kwargs["target_pos"]
        return valid, {"x": zero, "y": zero, "z": zero}, {
            "tx_pos": point,
            "first_hit": point,
            "last_hit": point,
        }

    monkeypatch.setattr(path_export, "epc_reflection_chain_to_target", fake_epc)

    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            num_samples=64,
            max_bounces=1,
            max_diffraction_order=0,
            max_num_paths=4,
            tuning=wc.path.Tuning()
            if backend is None
            else wc.path.Tuning(reflection_field_backend=backend),
        ),
    )

    assert epc_calls
    assert set(epc_calls) == {(expected_prefer, expected_require)}
    reflection_backend = result.metadata["runtime_backends"]["reflection"]
    assert reflection_backend["epc_backend"] == expected_metadata_backend
    assert reflection_backend["rayd_epc_required"] is expected_require


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


@pytest.mark.gpu
def test_path_solve_rayd_diffraction_path_export_metadata(monkeypatch):
    from witwin.channel.deterministic.trace import path_export
    from witwin.channel.path import solver as path_solver

    scene = wc.Scene(
        transmitters=[wc.Transmitter(name="tx", position=(0.0, 0.0, 1.0))],
        receivers=[wc.Receiver(name="rx", position=(0.0, 0.0, -1.0))],
        frequency=3.5e9,
        device="cuda",
    )
    native_calls = []

    def fake_los(**_kwargs):
        return path_export.empty_raw_paths(depth=1, return_geometry=False)

    def fake_reflections(**kwargs):
        return (
            (
                path_export.empty_raw_paths(
                    depth=1, return_geometry=kwargs["return_geometry"],
                ),
            ),
            (None,),
        )

    def fake_diffraction_states(**kwargs):
        runtime = kwargs["runtime"]
        return [
            {
                "runtime": runtime,
                "receiver_index_map": wt.UInt32([0]),
                "rx_positions": runtime.rx.positions,
                "state_arrays": {"n_states": 1, "__path_export_state_layout__": "reduced_v2"},
                "edge_data": None,
                "builder_report": {},
            }
        ]

    def forbidden_drijit_diffraction(**_kwargs):
        raise AssertionError("explicit rayd_optix path export must bypass DrJit diffraction materialization")

    def fake_native_paths(**kwargs):
        native_calls.append(kwargs)
        return SimpleNamespace(
            capacity=1,
            count=wt.Int32([1]),
            valid=wt.Bool([True]),
            tx_id=wt.Int32([0]),
            rx_id=wt.Int32([0]),
            order=wt.Int32([1]),
            edge0=wt.Int32([0]),
            edge1=wt.Int32([-1]),
            edge2=wt.Int32([-1]),
            delay=wt.Float([2.0 / 299792458.0]),
            field_x=wt.Complex2f(wt.Float([0.5]), wt.Float([0.25])),
            field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
            field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
            p0=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0])),
            p1=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0])),
            p2=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0])),
        )

    monkeypatch.setattr(path_solver, "collect_los_paths", fake_los)
    monkeypatch.setattr(path_solver, "collect_reflection_paths_for_transmitters", fake_reflections)
    monkeypatch.setattr(path_solver, "trace_diffraction_raw_collections", fake_diffraction_states)
    monkeypatch.setattr(path_solver, "collect_diffraction_state_paths", forbidden_drijit_diffraction)
    monkeypatch.setattr(scene, "trace_dfr_paths", fake_native_paths)

    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            max_bounces=0,
            max_diffraction_order=1,
            max_num_paths=4,
            tuning=wc.path.Tuning(
                diffraction_execution={"accumulate_primal": "rayd_optix"},
            ),
        ),
    )

    assert len(native_calls) == 1
    assert native_calls[0]["return_geometry"] is False
    assert result.metadata["path_counts"]["diffraction"] == 1
    assert result.metadata["runtime_backends"]["diffraction"] == {
        "implementation": "rayd_trace_dfr_paths_order1",
        "path_export_backend": "rayd_optix_compact_paths",
        "max_order": 1,
    }


@pytest.mark.gpu
def test_path_solve_auto_diffraction_defaults_to_rayd_optix_path_export(monkeypatch):
    from witwin.channel.deterministic.trace import path_export
    from witwin.channel.path import solver as path_solver

    scene = wc.Scene(
        transmitters=[wc.Transmitter(name="tx", position=(0.0, 0.0, 1.0))],
        receivers=[wc.Receiver(name="rx", position=(0.0, 0.0, -1.0))],
        frequency=3.5e9,
        device="cuda",
    )
    native_calls = []

    def fake_los(**_kwargs):
        return path_export.empty_raw_paths(depth=1, return_geometry=False)

    def fake_reflections(**kwargs):
        return (
            (
                path_export.empty_raw_paths(
                    depth=1, return_geometry=kwargs["return_geometry"],
                ),
            ),
            (None,),
        )

    def fake_diffraction_states(**kwargs):
        runtime = kwargs["runtime"]
        return [
            {
                "runtime": runtime,
                "receiver_index_map": wt.UInt32([0]),
                "rx_positions": runtime.rx.positions,
                "state_arrays": {"n_states": 1, "__path_export_state_layout__": "reduced_v2"},
                "edge_data": None,
                "builder_report": {},
            }
        ]

    def forbidden_drijit_diffraction(**_kwargs):
        raise AssertionError("auto path diffraction must resolve to RayD OptiX for order-1 workloads")

    def fake_native_paths(**kwargs):
        native_calls.append(kwargs)
        return SimpleNamespace(
            capacity=1,
            count=wt.Int32([1]),
            valid=wt.Bool([True]),
            tx_id=wt.Int32([0]),
            rx_id=wt.Int32([0]),
            order=wt.Int32([1]),
            edge0=wt.Int32([0]),
            edge1=wt.Int32([-1]),
            edge2=wt.Int32([-1]),
            delay=wt.Float([2.0 / 299792458.0]),
            field_x=wt.Complex2f(wt.Float([0.5]), wt.Float([0.25])),
            field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
            field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
            p0=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0])),
            p1=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0])),
            p2=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0])),
        )

    monkeypatch.setattr(path_solver, "collect_los_paths", fake_los)
    monkeypatch.setattr(path_solver, "collect_reflection_paths_for_transmitters", fake_reflections)
    monkeypatch.setattr(path_solver, "trace_diffraction_raw_collections", fake_diffraction_states)
    monkeypatch.setattr(path_solver, "collect_diffraction_state_paths", forbidden_drijit_diffraction)
    monkeypatch.setattr(scene, "trace_dfr_paths", fake_native_paths)

    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            max_bounces=0,
            max_diffraction_order=1,
            max_num_paths=4,
        ),
    )

    assert len(native_calls) == 1
    assert result.metadata["path_counts"]["diffraction"] == 1
    assert result.metadata["runtime_backends"]["diffraction"]["implementation"] == (
        "rayd_trace_dfr_paths_order1"
    )


@pytest.mark.gpu
def test_path_solve_auto_higher_order_diffraction_reports_drijit_backend(monkeypatch):
    from witwin.channel.deterministic.trace import path_export
    from witwin.channel.path import solver as path_solver

    scene = wc.Scene(
        transmitters=[wc.Transmitter(name="tx", position=(0.0, 0.0, 1.0))],
        receivers=[wc.Receiver(name="rx", position=(0.0, 0.0, -1.0))],
        frequency=3.5e9,
        device="cuda",
    )
    drjit_calls = []

    def fake_los(**_kwargs):
        return path_export.empty_raw_paths(depth=1, return_geometry=False)

    def fake_reflections(**kwargs):
        return (
            (
                path_export.empty_raw_paths(
                    depth=1, return_geometry=kwargs["return_geometry"],
                ),
            ),
            (None,),
        )

    def fake_diffraction_states(**kwargs):
        runtime = kwargs["runtime"]
        return [
            {
                "runtime": runtime,
                "receiver_index_map": wt.UInt32([0]),
                "rx_positions": runtime.rx.positions,
                "state_arrays": {"n_states": 0},
                "edge_data": None,
                "builder_report": {},
            }
        ]

    def fake_drijit_paths(**kwargs):
        drjit_calls.append(kwargs)
        return path_export.empty_raw_paths(depth=2, return_geometry=kwargs["return_geometry"])

    def forbidden_native_paths(**_kwargs):
        raise AssertionError("auto higher-order path diffraction must not use first-order RayD export")

    monkeypatch.setattr(path_solver, "collect_los_paths", fake_los)
    monkeypatch.setattr(path_solver, "collect_reflection_paths_for_transmitters", fake_reflections)
    monkeypatch.setattr(path_solver, "trace_diffraction_raw_collections", fake_diffraction_states)
    monkeypatch.setattr(path_solver, "collect_diffraction_state_paths", fake_drijit_paths)
    monkeypatch.setattr(scene, "trace_dfr_paths", forbidden_native_paths)

    result = wc.path.solve(
        scene=scene,
        transmitter="tx",
        receiver="rx",
        config=wc.path.Config(
            max_bounces=0,
            max_diffraction_order=2,
            max_num_paths=4,
        ),
    )

    assert len(drjit_calls) == 1
    assert result.metadata["runtime_backends"]["diffraction"] == {
        "implementation": "drjit_diffraction_state_path_export",
        "path_export_backend": "drjit_state_materialization",
        "max_order": 2,
    }


@pytest.mark.gpu
def test_path_solve_rayd_diffraction_path_export_smoke_stable_counts():
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
    config = wc.path.Config(
        num_samples=64,
        max_bounces=0,
        max_diffraction_order=1,
        max_num_paths=8,
        return_geometry=True,
        edge_policy=wc.EdgePolicy(edge_selection_mode="all_edges"),
        tuning=wc.path.Tuning(
            diffraction_execution={"accumulate_primal": "rayd_optix"},
        ),
    )

    first = wc.path.solve(scene=scene, transmitter="tx", receiver="rx", config=config)
    second = wc.path.solve(scene=scene, transmitter="tx", receiver="rx", config=config)

    first_count = int(first.metadata["path_counts"]["diffraction"])
    second_count = int(second.metadata["path_counts"]["diffraction"])
    assert first_count == second_count
    assert first_count > 0
    assert first.metadata["runtime_backends"]["diffraction"]["implementation"] == (
        "rayd_trace_dfr_paths_order1"
    )
    assert first.vertices is not None
    a = np.asarray(first.a)
    tau = np.asarray(first.tau)
    assert np.isfinite(a.real).any()
    assert np.isfinite(a.imag).any()
    assert np.isfinite(tau).any()
