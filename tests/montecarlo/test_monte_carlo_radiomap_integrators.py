"""Non-GPU contract checks for standalone Monte Carlo radiomap integrators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import drjit as dr
import numpy as np

from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.grid import Grid, GridSpec
from witwin.channel.montecarlo import (
    ComponentFilterConfig,
    Config,
    DiffractionExecutionConfig,
    FilterConfig,
    IntegratorOptions,
    Tuning,
)
from witwin.channel.montecarlo import types as wt
from witwin.channel.montecarlo.integrators import (
    BDPT,
    Basic,
    Integrator,
)
from witwin.channel.montecarlo.integrators import basic as basic_integrator
from witwin.channel.montecarlo.integrators import bdpt as bdpt_integrator
from witwin.channel.montecarlo.integrators.bdpt_diffraction import BDPTDiffractionMIS, BDPTDiffractionResult
from witwin.channel.montecarlo.filtering import apply_component_filter, apply_power_filtering
from witwin.channel.montecarlo.integrators.basic import _empty_radio_map, finalize_weighted_diagnostics
from witwin.channel.montecarlo.integrators.basic_ad import BasicIntegratorAD
from witwin.channel.montecarlo.integrators.metadata import build_metadata
from witwin.channel.montecarlo.grid_ops import GridContributionStore
from witwin.channel.core.physics.materials import FaceMaterial
from witwin.channel.core.scene import Scene
from witwin.channel.montecarlo.trace.diffraction import (
    Diffraction,
    DiffractionEdgeSampler,
    DiffractionStates,
)


def test_config_integrator_contract_is_explicit():
    assert Config().integrator_options.integrator == "basic"
    assert Config(integrator_options=IntegratorOptions(integrator="bdpt")).integrator_options.integrator == "bdpt"
    assert Config(
        integrator_options=IntegratorOptions(integrator="bdpt"),
    ).integrator_options.bdpt_diffraction_sampling == "sobol"
    assert Config(
        integrator_options=IntegratorOptions(integrator="bdpt", bdpt_diffraction_sampling="hash"),
    ).integrator_options.bdpt_diffraction_sampling == "hash"
    assert Config(
        integrator_options=IntegratorOptions(integrator="bdpt"),
        max_diffraction_order=3,
    ).max_diffraction_order == 3
    assert Config(
        max_diffraction_order=2,
        integrator_options=IntegratorOptions(integrator="bdpt"),
        tuning=Tuning(enable_bdpt_reflection_coupled_diffraction=False),
    ).tuning.enable_bdpt_reflection_coupled_diffraction is False

    with pytest.raises(ValueError, match="integrator"):
        Config(integrator_options=IntegratorOptions(integrator="specular"))

    with pytest.raises(ValueError, match="bdpt_diffraction_sampling"):
        Config(integrator_options=IntegratorOptions(integrator="bdpt", bdpt_diffraction_sampling="white_noise"))

    with pytest.raises(ValueError, match="basic"):
        Config(max_diffraction_order=3)


def test_config_accepts_explicit_rayd_reflection_accumulation_backend():
    options = IntegratorOptions(accumulation_backend="rayd_reflection_accumulation")

    assert options.accumulation_backend == "rayd_reflection_accumulation"


def test_accumulation_auto_can_prefer_rayd_for_primal(monkeypatch):
    monkeypatch.setattr(
        basic_integrator.NativeExtension,
        "native_extension_available",
        staticmethod(lambda: True),
    )

    assert Basic.resolve_accumulation("auto") == "native_monte_carlo"
    assert (
        Basic.resolve_accumulation("auto", prefer_rayd=True)
        == "rayd_reflection_accumulation"
    )
    assert (
        Basic.resolve_accumulation("native_monte_carlo", prefer_rayd=True)
        == "native_monte_carlo"
    )
    assert (
        Basic.resolve_accumulation("rayd_reflection_accumulation")
        == "rayd_reflection_accumulation"
    )


def test_rayd_accumulation_prepare_config_uses_compact_sampling_metadata(monkeypatch):
    monkeypatch.setattr(
        basic_integrator.NativeExtension,
        "native_extension_available",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        basic_integrator,
        "_capture_free_bytes",
        lambda **_kwargs: 64 * 1024 * 1024,
    )

    def forbidden_metadata(**_kwargs):
        raise AssertionError("RayD accumulation should not build DrJit sampling metadata")

    monkeypatch.setattr(basic_integrator.Sampler, "metadata", staticmethod(forbidden_metadata))
    setup = Basic.prepare_config(
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        grid_spec=_filter_test_grid(nx=1, ny=1),
        mc_config=Config(),
        scene=SimpleNamespace(),
        config=SimpleNamespace(cell_size=1.0, wavelength=0.125),
        tx_power=1.0,
        accumulation_backend="auto",
        return_timing=False,
        loop_mode="symbolic",
        effective={"reflection_n_rays": 8, "max_diffractions": 0},
        prefer_rayd_accumulation=True,
    )

    assert setup.resolved_accumulation_backend == "rayd_reflection_accumulation"
    assert setup.ray_sampling_metadata == {
        "requested_ray_sampling": "full_sphere",
        "selected_ray_sampling": "full_sphere",
        "sampling_sequence": "rayd_native_sampler",
    }
    assert setup.reflection_solid_angle_per_ray == pytest.approx(4.0 * np.pi / 8.0)


def test_diffraction_execution_config_accepts_rayd_optix_accumulation():
    config = Config(
        tuning=Tuning(
            diffraction_execution={
                "accumulate_primal": "rayd_optix",
            },
        )
    ).to_trace_config()

    assert config.diffraction_execution.accumulate_primal == "rayd_optix"
    assert DiffractionExecutionConfig(accumulate_primal="auto").accumulate_primal == "auto"

    with pytest.raises(ValueError, match="accumulate_primal"):
        DiffractionExecutionConfig(accumulate_primal="cuda")


def test_diffraction_auto_resolves_to_rayd_for_primal_and_ad_tapes():
    config = Config().to_trace_config()

    assert Basic._diffraction_accumulate_primal_mode(
        config,
        collect_ad_tapes=False,
    ) == "rayd_optix"
    assert BDPTDiffractionMIS._diffraction_accumulate_primal_mode(
        config,
        collect_ad_tapes=False,
    ) == "rayd_optix"
    assert Basic._diffraction_accumulate_primal_mode(
        config,
        collect_ad_tapes=True,
    ) == "rayd_optix"
    assert BDPTDiffractionMIS._diffraction_accumulate_primal_mode(
        config,
        collect_ad_tapes=True,
    ) == "rayd_optix"


def test_diffraction_states_convert_to_rayd_state_table():
    states = DiffractionStates(
        edge_index=wt.Int32([7, 8]),
        edge_pos=wt.Point3f(
            wt.Float([0.0, 1.0]),
            wt.Float([0.0, 3.0]),
            wt.Float([2.0, 1.0]),
        ),
        edge_dir=wt.Vector3f(
            wt.Float([1.0, 0.0]),
            wt.Float([0.0, 1.0]),
            wt.Float([0.0, 0.0]),
        ),
        n0=wt.Vector3f(
            wt.Float([0.0, 1.0]),
            wt.Float([1.0, 0.0]),
            wt.Float([0.0, 0.0]),
        ),
        nn=wt.Vector3f(
            wt.Float([0.0, -1.0]),
            wt.Float([-1.0, 0.0]),
            wt.Float([0.0, 0.0]),
        ),
        wedge_n=wt.Float([1.25, 1.5]),
        edge_line_min=wt.Float([-0.5, -0.25]),
        edge_line_max=wt.Float([0.5, 0.25]),
        source_pos=wt.Point3f(
            wt.Float([0.0, 1.0]),
            wt.Float([0.0, 1.0]),
            wt.Float([0.0, 1.0]),
        ),
        adjacent_face0=wt.Int32([11, 12]),
        adjacent_face1=wt.Int32([13, 14]),
        face0_material=FaceMaterial(
            eta_r=wt.Float([4.0, 5.0]),
            sigma=wt.Float([0.01, 0.02]),
            gain=wt.Float([1.0, 1.0]),
            use_fresnel=wt.Bool([True, True]),
            mu_r=wt.Float([1.0, 1.1]),
        ),
        face1_material=FaceMaterial(
            eta_r=wt.Float([6.0, 7.0]),
            sigma=wt.Float([0.03, 0.04]),
            gain=wt.Float([1.0, 1.0]),
            use_fresnel=wt.Bool([True, False]),
            mu_r=wt.Float([1.2, 1.3]),
        ),
        source_power=wt.Float([0.75, 0.25]),
        prefix_reflection_depth=wt.Int32([0, 2]),
        prefix_initial_ray_dir=wt.Vector3f(
            wt.Float([9.0, -1.0]),
            wt.Float([9.0, 0.0]),
            wt.Float([9.0, 0.0]),
        ),
    )

    table = states._to_rayd_state_table(Scene)
    dr.eval(
        table.edge_index,
        table.edge_pos,
        table.edge_dir,
        table.n0,
        table.n1,
        table.prim0,
        table.prim1,
        table.exterior_angle,
        table.src,
        table.src_power,
        table.wi,
        table.d0,
        table.prefix_depth,
    )

    assert table.count == 2
    np.testing.assert_array_equal(
        np.asarray(table.edge_index, dtype=np.int32),
        np.array([7, 8], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(table.edge_pos.y, dtype=np.float32),
        np.array([0.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(table.edge_dir.x, dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(table.n0.y, dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(table.n1.x, dtype=np.float32),
        np.array([0.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(table.prim0, dtype=np.int32),
        np.array([11, 12], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(table.prim1, dtype=np.int32),
        np.array([13, 14], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(table.exterior_angle, dtype=np.float32),
        np.array([1.25 * np.pi, 1.5 * np.pi], dtype=np.float32),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(table.src.z, dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(table.src_power, dtype=np.float32),
        np.array([0.75, 0.25], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(table.wi.z, dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(table.d0.x, dtype=np.float32),
        np.array([0.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(table.prefix_depth, dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
    )


def test_config_filtering_contract_is_explicit():
    assert Config().filtering is None

    cfg = Config(
        filtering=FilterConfig(
            reflection=ComponentFilterConfig(
                method="gaussian",
                radius=1,
                sigma=1.0,
                blend=0.25,
            ),
            diffraction={
                "method": "bilateral",
                "radius": 2,
                "sigma": 1.25,
                "range_sigma": 0.5,
            },
        )
    )

    assert isinstance(cfg.filtering, FilterConfig)
    assert isinstance(cfg.filtering.reflection, ComponentFilterConfig)
    assert isinstance(cfg.filtering.diffraction, ComponentFilterConfig)
    assert cfg.filtering.reflection.method == "gaussian"
    assert cfg.filtering.reflection.radius == 1
    assert cfg.filtering.reflection.sigma == pytest.approx(1.0)
    assert cfg.filtering.reflection.blend == pytest.approx(0.25)
    assert cfg.filtering.diffraction.method == "bilateral"
    assert cfg.filtering.diffraction.range_sigma == pytest.approx(0.5)

    mapped = Config(
        filtering={
            "reflection": {"method": "gaussian", "radius": 0, "sigma": 1.0},
        }
    )
    assert mapped.filtering.reflection.radius == 0
    assert mapped.filtering.diffraction is None

    with pytest.raises(ValueError, match="method"):
        ComponentFilterConfig(method="box")
    with pytest.raises(ValueError, match="radius"):
        ComponentFilterConfig(radius=-1)
    with pytest.raises(ValueError, match="sigma"):
        ComponentFilterConfig(sigma=0.0)
    with pytest.raises(ValueError, match="range_sigma"):
        ComponentFilterConfig(method="bilateral", range_sigma=0.0)
    with pytest.raises(ValueError, match="blend"):
        ComponentFilterConfig(blend=1.25)


def test_integrator_exports_are_explicit():
    assert isinstance(Basic(), Integrator)
    assert isinstance(BDPT(), Integrator)
    assert Basic().mode == "basic"
    assert BDPT().mode == "bdpt"


def _filter_test_grid(nx: int = 3, ny: int = 3):
    return Grid.from_spec(
        GridSpec(
            axis="z",
            position=0.0,
            bounds=((0.0, float(nx)), (0.0, float(ny))),
            grid_shape=(nx, ny),
        )
    )


def _single_diffraction_states(*, prefix_depth: int = 0):
    return DiffractionStates(
        edge_index=wt.Int32([0]),
        edge_pos=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([0.0])),
        edge_dir=wt.Vector3f(wt.Float([1.0]), wt.Float([0.0]), wt.Float([0.0])),
        n0=wt.Vector3f(wt.Float([0.0]), wt.Float([1.0]), wt.Float([0.0])),
        nn=wt.Vector3f(wt.Float([0.0]), wt.Float([-1.0]), wt.Float([0.0])),
        wedge_n=wt.Float([1.5]),
        edge_line_min=wt.Float([-0.5]),
        edge_line_max=wt.Float([0.5]),
        source_pos=wt.Point3f(wt.Float([0.0]), wt.Float([0.0]), wt.Float([1.0])),
        adjacent_face0=wt.Int32([0]),
        adjacent_face1=wt.Int32([0]),
        face0_material=FaceMaterial(
            eta_r=wt.Float([4.0]),
            sigma=wt.Float([0.0]),
            gain=wt.Float([1.0]),
            use_fresnel=wt.Bool([True]),
            mu_r=wt.Float([1.0]),
        ),
        face1_material=FaceMaterial(
            eta_r=wt.Float([4.0]),
            sigma=wt.Float([0.0]),
            gain=wt.Float([1.0]),
            use_fresnel=wt.Bool([True]),
            mu_r=wt.Float([1.0]),
        ),
        source_power=wt.Float([1.0]),
        prefix_reflection_depth=wt.Int32([prefix_depth]),
        prefix_initial_ray_dir=wt.Vector3f(
            wt.Float([0.0]),
            wt.Float([0.0]),
            wt.Float([-1.0]),
        ),
    )


def _two_recursive_diffraction_states():
    return DiffractionStates(
        edge_index=wt.Int32([1, 2]),
        edge_pos=wt.Point3f(
            wt.Float([0.0, 0.0]),
            wt.Float([0.5, 1.0]),
            wt.Float([0.0, 0.0]),
        ),
        edge_dir=wt.Vector3f(
            wt.Float([1.0, 1.0]),
            wt.Float([0.0, 0.0]),
            wt.Float([0.0, 0.0]),
        ),
        n0=wt.Vector3f(
            wt.Float([0.0, 0.0]),
            wt.Float([1.0, 1.0]),
            wt.Float([0.0, 0.0]),
        ),
        nn=wt.Vector3f(
            wt.Float([0.0, 0.0]),
            wt.Float([-1.0, -1.0]),
            wt.Float([0.0, 0.0]),
        ),
        wedge_n=wt.Float([1.5, 1.5]),
        edge_line_min=wt.Float([-0.5, -0.5]),
        edge_line_max=wt.Float([0.5, 0.5]),
        source_pos=wt.Point3f(
            wt.Float([0.0, 0.0]),
            wt.Float([0.0, 0.0]),
            wt.Float([1.0, 1.0]),
        ),
        adjacent_face0=wt.Int32([0, 0]),
        adjacent_face1=wt.Int32([0, 0]),
        face0_material=FaceMaterial(
            eta_r=wt.Float([4.0, 4.0]),
            sigma=wt.Float([0.0, 0.0]),
            gain=wt.Float([1.0, 1.0]),
            use_fresnel=wt.Bool([True, True]),
            mu_r=wt.Float([1.0, 1.0]),
        ),
        face1_material=FaceMaterial(
            eta_r=wt.Float([4.0, 4.0]),
            sigma=wt.Float([0.0, 0.0]),
            gain=wt.Float([1.0, 1.0]),
            use_fresnel=wt.Bool([True, True]),
            mu_r=wt.Float([1.0, 1.0]),
        ),
        source_power=wt.Float([1.0, 1.0]),
        prefix_reflection_depth=wt.Int32([0, 0]),
        prefix_initial_ray_dir=wt.Vector3f(
            wt.Float([0.0, 0.0]),
            wt.Float([0.0, 0.0]),
            wt.Float([-1.0, -1.0]),
        ),
    )


def test_scene_accum_dfr_builds_rayd_options():
    class FakeRayDScene:
        def __init__(self):
            self.calls = []

        def accum_dfr(
            self,
            initial_table,
            recursive_table,
            grid_desc,
            material,
            options,
            active_mask,
        ):
            self.calls.append(
                {
                    "initial_count": int(initial_table.count),
                    "recursive_count": int(recursive_table.count),
                    "grid_resolution0": int(grid_desc.resolution0),
                    "grid_resolution1": int(grid_desc.resolution1),
                    "material_width": int(dr.width(material.eta_r)),
                    "options": options,
                    "active_width": int(dr.width(active_mask)),
                }
            )
            return SimpleNamespace(power=wt.Float([0.0]))

    fake_rayd = FakeRayDScene()
    scene = Scene.__new__(Scene)
    scene._rayd_scene = fake_rayd
    scene._merged_vertices = lambda: None
    scene._triangle_runtime = lambda: {
        "n_triangles": 1,
        "material_eps_r": wt.Float([4.0]),
        "material_sigma_e": wt.Float([0.0]),
        "material_mu_r": wt.Float([1.0]),
        "material_specified": wt.Bool([True]),
    }

    result = scene.accum_dfr(
        initial_states=_single_diffraction_states(),
        recursive_states=_single_diffraction_states(),
        grid=_filter_test_grid(nx=1, ny=1),
        config=SimpleNamespace(wavelength=0.125, k=50.0),
        seed=23,
        samples=10,
        direct_samples=3,
        keller_samples=2,
        max_order=2,
        sample_sequence="hash",
        active=True,
    )

    assert int(dr.width(result.power)) == 1
    assert len(fake_rayd.calls) == 1
    call = fake_rayd.calls[0]
    assert call["initial_count"] == 1
    assert call["recursive_count"] == 1
    assert call["grid_resolution0"] == 1
    assert call["grid_resolution1"] == 1
    assert call["material_width"] == 1
    assert call["active_width"] == 1
    assert call["options"].max_order == 2
    assert call["options"].direct_samples == 3
    assert call["options"].keller_samples == 2
    assert call["options"].strategy_mask != 0


def test_scene_accum_dfr_accepts_order3_options():
    class FakeRayDScene:
        def __init__(self):
            self.calls = []

        def accum_dfr(
            self,
            initial_table,
            recursive_table,
            grid_desc,
            material,
            options,
            active_mask,
        ):
            self.calls.append(
                {
                    "initial_count": int(initial_table.count),
                    "recursive_count": int(recursive_table.count),
                    "options": options,
                    "active_width": int(dr.width(active_mask)),
                }
            )
            return SimpleNamespace(power=wt.Float([0.0]))

    fake_rayd = FakeRayDScene()
    scene = Scene.__new__(Scene)
    scene._rayd_scene = fake_rayd
    scene._merged_vertices = lambda: None
    scene._triangle_runtime = lambda: {
        "n_triangles": 1,
        "material_eps_r": wt.Float([4.0]),
        "material_sigma_e": wt.Float([0.0]),
        "material_mu_r": wt.Float([1.0]),
        "material_specified": wt.Bool([True]),
    }

    scene.accum_dfr(
        initial_states=_single_diffraction_states(),
        recursive_states=_two_recursive_diffraction_states(),
        grid=_filter_test_grid(nx=1, ny=1),
        config=SimpleNamespace(wavelength=0.125, k=50.0),
        seed=29,
        samples=12,
        direct_samples=4,
        keller_samples=3,
        max_order=3,
        sample_sequence="hash",
        active=True,
    )

    assert len(fake_rayd.calls) == 1
    call = fake_rayd.calls[0]
    assert call["initial_count"] == 1
    assert call["recursive_count"] == 2
    assert call["active_width"] == 1
    assert call["options"].max_order == 3
    assert call["options"].direct_samples == 4
    assert call["options"].keller_samples == 3


def test_grid_contribution_store_accumulates_directly_without_replay():
    grid = _filter_test_grid(nx=2, ny=2)
    diagnostics = _empty_radio_map(grid.n_cells)
    store = GridContributionStore(
        capacity=3,
        grid=grid,
        weighted_diagnostics=diagnostics,
    )

    store.store(
        coord_0=wt.Float([0.25, 1.25, 0.75]),
        coord_1=wt.Float([0.25, 0.25, 1.25]),
        component_power={
            "reflection": wt.Float([1.0, 2.0, 4.0]),
            "diffraction": wt.Float([0.5, 0.25, 8.0]),
        },
        active=wt.Bool([True, True, False]),
    )

    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["reflection"], dtype=np.float32),
        np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.array([0.5, 0.25, 0.0, 0.0], dtype=np.float32),
    )
    assert scalar(store.next_slot) == 2

    store.scatter_into(grid=grid, weighted_diagnostics=diagnostics)

    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["reflection"], dtype=np.float32),
        np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
    )


def test_grid_contribution_store_direct_power_is_differentiable():
    grid = _filter_test_grid(nx=2, ny=2)
    diagnostics = _empty_radio_map(grid.n_cells)
    store = GridContributionStore(
        capacity=2,
        grid=grid,
        weighted_diagnostics=diagnostics,
    )
    power = wt.Float([1.0, 2.0])
    dr.enable_grad(power)

    store.store(
        coord_0=wt.Float([0.25, 1.25]),
        coord_1=wt.Float([0.25, 0.25]),
        component_power={"reflection": power},
        active=wt.Bool([True, True]),
    )

    dr.set_grad(power, wt.Float([3.0, 5.0]))
    jvp = dr.forward_to(
        diagnostics["incoherent"]["reflection"],
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )

    np.testing.assert_allclose(
        np.asarray(jvp, dtype=np.float32),
        np.array([3.0, 5.0, 0.0, 0.0], dtype=np.float32),
    )


def test_basic_reflection_uses_single_symbolic_batch(monkeypatch):
    calls = []

    def trace_stub(**kwargs):
        ray_index = kwargs["ray_index"]
        calls.append(
            {
                "width": int(dr.width(ray_index)),
                "indices": np.asarray(ray_index, dtype=np.uint32).tolist(),
            }
        )
        return wt.UInt32(1), wt.UInt32(2), kwargs["diff_state_store"]

    monkeypatch.setattr(
        basic_integrator.mc_reflection.Reflection,
        "trace",
        staticmethod(trace_stub),
    )
    grid = _filter_test_grid(nx=2, ny=2)
    diagnostics = _empty_radio_map(grid.n_cells)

    result = Basic.run_reflection(
        scene=object(),
        grid=grid,
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=object(),
        samples_per_tx=5,
        seed=3,
        ray_batch_size=2,
        solid_angle_per_ray=1.0,
        cell_area=1.0,
        effective={"reflection_max_bounces": 1},
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold_linear=0.0,
        material_omega=wt.Float(0.0),
        weighted_diagnostics=diagnostics,
        collect_wedges=False,
        collect_ad_tapes=False,
        loop_mode="symbolic",
        resolved_accumulation_backend="native_monte_carlo",
    )

    assert calls == [{"width": 5, "indices": [0, 1, 2, 3, 4]}]
    assert scalar(result.path_counts.los) == 1
    assert scalar(result.path_counts.reflection) == 2


def test_basic_reflection_rayd_accumulation_scatter_and_wedges(monkeypatch):
    def forbidden_trace(**_kwargs):
        raise AssertionError("rayd reflection accumulation should bypass Reflection.trace")

    monkeypatch.setattr(
        basic_integrator.mc_reflection.Reflection,
        "trace",
        staticmethod(forbidden_trace),
    )

    class RayDAccumulationScene:
        def __init__(self):
            self.calls = []

        def accumulate_reflections(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                reflection_power=wt.Float([0.25, 0.75, 0.0, 0.0]),
                reflection_field_x=wt.Complex2f(
                    wt.Float([0.5, 0.0, 0.0, 0.0]),
                    wt.Float([0.0, 0.0, 0.0, 0.0]),
                ),
                reflection_field_y=wt.Complex2f(
                    wt.Float([0.0, 1.0, 0.0, 0.0]),
                    wt.Float([0.0, 0.0, 0.0, 0.0]),
                ),
                reflection_field_z=wt.Complex2f(
                    wt.Float([0.0, 0.0, 0.0, 0.0]),
                    wt.Float([0.0, 0.0, 0.0, 0.0]),
                ),
                reflection_count=wt.Int32([2]),
                wedge_events=SimpleNamespace(
                    capacity=2,
                    count=wt.Int32([1]),
                    ray_index=wt.Int32([0, -1]),
                    hit_points=wt.Point3f(
                        wt.Float([0.5, 0.0]),
                        wt.Float([0.5, 0.0]),
                        wt.Float([0.0, 0.0]),
                    ),
                    normals=wt.Vector3f(
                        wt.Float([0.0, 0.0]),
                        wt.Float([0.0, 0.0]),
                        wt.Float([1.0, 0.0]),
                    ),
                    prim_id=wt.Int32([3, -1]),
                    directions=wt.Vector3f(
                        wt.Float([0.0, 0.0]),
                        wt.Float([0.0, 0.0]),
                        wt.Float([-1.0, 0.0]),
                    ),
                    bounce_depth=wt.Int32([0, -1]),
                ),
            )

    scene = RayDAccumulationScene()
    grid = _filter_test_grid(nx=2, ny=2)
    diagnostics = _empty_radio_map(grid.n_cells)

    result = Basic.run_reflection(
        scene=scene,
        grid=grid,
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            tx_polarization=(0.0, 1.0, 0.0),
            rx_polarization=(0.0, 1.0, 0.0),
        ),
        samples_per_tx=2,
        seed=3,
        ray_batch_size=2,
        solid_angle_per_ray=0.5,
        cell_area=1.0,
        effective={"reflection_max_bounces": 1},
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold_linear=0.0,
        material_omega=wt.Float(0.0),
        weighted_diagnostics=diagnostics,
        collect_wedges=True,
        collect_ad_tapes=False,
        loop_mode="symbolic",
        resolved_accumulation_backend="rayd_reflection_accumulation",
    )

    assert len(scene.calls) == 1
    assert scene.calls[0]["collect_wedges"] is True
    assert scene.calls[0]["tx_polarization"] == (0.0, 1.0, 0.0)
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["reflection"], dtype=np.float32),
        np.array([0.25, 0.75, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics["coherent"]["reflection"].real, dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics["coherent_power"]["reflection"], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    assert scalar(result.path_counts.reflection) == 2
    assert result.diff_state_store is not None
    finalized = result.diff_state_store.finalize()
    assert finalized["count"] == 1
    assert np.asarray(finalized["prim_index"], dtype=np.int32).tolist() == [3]
    np.testing.assert_allclose(
        np.asarray(finalized["source_power"], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )


def test_basic_reflection_rayd_accumulation_accepts_ad_tape_collection(monkeypatch):
    def forbidden_trace(**_kwargs):
        raise AssertionError("rayd reflection accumulation should bypass Reflection.trace")

    monkeypatch.setattr(
        basic_integrator.mc_reflection.Reflection,
        "trace",
        staticmethod(forbidden_trace),
    )

    class RayDAccumulationScene:
        def __init__(self):
            self.calls = []

        def accumulate_reflections(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                reflection_power=wt.Float([0.25]),
                reflection_field_x=wt.Complex2f(wt.Float([0.5]), wt.Float([0.0])),
                reflection_field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_count=wt.Int32([1]),
                wedge_events=SimpleNamespace(capacity=0, count=wt.Int32([0])),
            )

    scene = RayDAccumulationScene()
    diagnostics = _empty_radio_map(1)

    result = Basic.run_reflection(
        scene=scene,
        grid=_filter_test_grid(nx=1, ny=1),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            tx_polarization=(1.0, 0.0, 0.0),
            rx_polarization=(1.0, 0.0, 0.0),
        ),
        samples_per_tx=1,
        seed=3,
        ray_batch_size=1,
        solid_angle_per_ray=1.0,
        cell_area=1.0,
        effective={"reflection_max_bounces": 1},
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold_linear=0.0,
        material_omega=wt.Float(0.0),
        weighted_diagnostics=diagnostics,
        collect_wedges=False,
        collect_ad_tapes=True,
        loop_mode="symbolic",
        resolved_accumulation_backend="rayd_reflection_accumulation",
    )

    assert scene.calls
    assert scalar(result.path_counts.reflection) == 1
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["reflection"], dtype=np.float32),
        np.array([0.25], dtype=np.float32),
    )

def test_basic_reflection_rayd_accumulation_forwards_prefix_wedges(monkeypatch):
    def forbidden_trace(**_kwargs):
        raise AssertionError("rayd reflection accumulation should bypass Reflection.trace")

    monkeypatch.setattr(
        basic_integrator.mc_reflection.Reflection,
        "trace",
        staticmethod(forbidden_trace),
    )

    class RayDAccumulationScene:
        def __init__(self):
            self.calls = []

        def accumulate_reflections(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                reflection_power=wt.Float([0.0]),
                reflection_field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_count=wt.Int32([0]),
                wedge_events=SimpleNamespace(
                    capacity=3,
                    count=wt.Int32([2]),
                    ray_index=wt.Int32([0, 0, -1]),
                    hit_points=wt.Point3f(
                        wt.Float([0.5, 0.25, 0.0]),
                        wt.Float([0.5, 0.75, 0.0]),
                        wt.Float([0.0, 0.0, 0.0]),
                    ),
                    normals=wt.Vector3f(
                        wt.Float([0.0, 0.0, 0.0]),
                        wt.Float([0.0, 0.0, 0.0]),
                        wt.Float([1.0, 1.0, 0.0]),
                    ),
                    prim_id=wt.Int32([3, 4, -1]),
                    directions=wt.Vector3f(
                        wt.Float([0.0, 0.0, 0.0]),
                        wt.Float([0.0, 1.0, 0.0]),
                        wt.Float([-1.0, 0.0, 0.0]),
                    ),
                    source_points=wt.Point3f(
                        wt.Float([0.0, 1.0, 0.0]),
                        wt.Float([0.0, 2.0, 0.0]),
                        wt.Float([1.0, 3.0, 0.0]),
                    ),
                    src_power=wt.Float([1.0, 0.25, 0.0]),
                    initial_directions=wt.Vector3f(
                        wt.Float([0.0, 0.0, 0.0]),
                        wt.Float([0.0, 0.0, 0.0]),
                        wt.Float([-1.0, -1.0, 0.0]),
                    ),
                    bounce_depth=wt.Int32([0, 1, -1]),
                ),
            )

    scene = RayDAccumulationScene()
    diagnostics = _empty_radio_map(1)

    result = Basic.run_reflection(
        scene=scene,
        grid=_filter_test_grid(nx=1, ny=1),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            tx_polarization=(1.0, 0.0, 0.0),
            rx_polarization=(1.0, 0.0, 0.0),
        ),
        samples_per_tx=2,
        seed=3,
        ray_batch_size=2,
        solid_angle_per_ray=0.5,
        cell_area=1.0,
        effective={"reflection_max_bounces": 2},
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold_linear=0.0,
        material_omega=wt.Float(0.0),
        weighted_diagnostics=diagnostics,
        collect_wedges=True,
        collect_ad_tapes=False,
        collect_wedge_prefixes=True,
        loop_mode="symbolic",
        resolved_accumulation_backend="rayd_reflection_accumulation",
    )

    assert scene.calls[0]["collect_wedge_prefixes"] is True
    assert scene.calls[0]["wedge_capacity"] == 6
    assert result.diff_state_store is not None
    finalized = result.diff_state_store.finalize()
    assert finalized["count"] == 2
    np.testing.assert_array_equal(
        np.asarray(finalized["prefix_reflection_depth"], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(finalized["source_power"], dtype=np.float32),
        np.array([1.0, 0.25], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(finalized["source_pos"].x, dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    )


def test_basic_reflection_rayd_accumulation_caps_prefix_wedge_collection(monkeypatch):
    def forbidden_trace(**_kwargs):
        raise AssertionError("rayd reflection accumulation should bypass Reflection.trace")

    monkeypatch.setattr(
        basic_integrator.mc_reflection.Reflection,
        "trace",
        staticmethod(forbidden_trace),
    )

    class RayDAccumulationScene:
        def __init__(self):
            self.calls = []

        def accumulate_reflections(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                reflection_power=wt.Float([0.0]),
                reflection_field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                reflection_count=wt.Int32([0]),
                wedge_events=SimpleNamespace(
                    capacity=2,
                    count=wt.Int32([0]),
                ),
            )

    scene = RayDAccumulationScene()
    diagnostics = _empty_radio_map(1)

    Basic.run_reflection(
        scene=scene,
        grid=_filter_test_grid(nx=1, ny=1),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(wavelength=0.125, k=50.0),
        samples_per_tx=5,
        seed=3,
        ray_batch_size=5,
        solid_angle_per_ray=0.2,
        cell_area=1.0,
        effective={"reflection_max_bounces": 2},
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold_linear=0.0,
        material_omega=wt.Float(0.0),
        weighted_diagnostics=diagnostics,
        collect_wedges=True,
        collect_ad_tapes=False,
        collect_wedge_prefixes=True,
        prefix_wedge_sample_cap=2,
        loop_mode="symbolic",
        resolved_accumulation_backend="rayd_reflection_accumulation",
    )

    assert scene.calls[0]["wedge_capacity"] == 2
    assert scene.calls[0]["wedge_sample_stride"] == 5


def test_basic_integrator_allows_rayd_accumulation_when_ad_requested(monkeypatch):
    class SceneWithoutGrad:
        def _triangle_runtime(self):
            return None

    calls = []

    monkeypatch.setattr(
        basic_integrator.SparseCoeffKernel,
        "available",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        BasicIntegratorAD,
        "integrate",
        staticmethod(lambda **kwargs: calls.append(kwargs) or "ad-result"),
    )

    config = Config(
        integrator_options=IntegratorOptions(
            ad=True,
            accumulation_backend="rayd_reflection_accumulation",
        )
    )

    result = Basic().integrate(
        wt.Point3f(0.0, 0.0, 1.0),
        _filter_test_grid(nx=1, ny=1),
        config,
        SceneWithoutGrad(),
        SimpleNamespace(),
        {"effective": {"reflection_max_bounces": 1, "max_diffractions": 0}},
        accumulation_backend="rayd_reflection_accumulation",
    )

    assert result == "ad-result"
    assert calls[0]["accumulation_backend"] == "rayd_reflection_accumulation"


def test_basic_diffraction_visibility_uses_segment_pair_queries():
    calls = []

    class PairOnlyScene:
        def segment_visible(self, *args, **kwargs):
            raise AssertionError("basic diffraction should use segment_pair_visible")

        def segment_pair_visible(self, start_pos, end_pos, end_pos_offset, *, active=True):
            calls.append(
                {
                    "start": start_pos,
                    "end": end_pos,
                    "end_offset": end_pos_offset,
                    "active": active,
                }
            )
            return wt.Bool([True, False])

    grid = _filter_test_grid(nx=2, ny=2)
    result = Diffraction.check_visibility(
        diff_point=wt.Point3f(
            wt.Float([0.5, 1.5]),
            wt.Float([0.5, 0.5]),
            wt.Float([0.0, 0.0]),
        ),
        diff_point_offset=wt.Point3f(
            wt.Float([0.55, 1.55]),
            wt.Float([0.5, 0.5]),
            wt.Float([0.0, 0.0]),
        ),
        source_pos=wt.Point3f(0.0, 0.5, 1.0),
        ko=wt.Vector3f(0.0, 0.0, -1.0),
        ray_origin=wt.Point3f(
            wt.Float([0.5, 1.5]),
            wt.Float([0.5, 0.5]),
            wt.Float([1.0, 1.0]),
        ),
        diffraction_batch_size=2,
        grid=grid,
        sample_active=wt.Bool([True, True]),
        scene=PairOnlyScene(),
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(
        np.asarray(result.source_visible, dtype=bool),
        np.array([True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(result.visible_target, dtype=bool),
        np.array([True, False]),
    )


def test_basic_diffraction_rayd_optix_accumulates_grid(monkeypatch):
    def forbidden_trace_batches(**_kwargs):
        raise AssertionError("rayd_optix diffraction should bypass DrJit trace_batches")

    monkeypatch.setattr(
        basic_integrator.Diffraction,
        "trace_batches",
        staticmethod(forbidden_trace_batches),
    )
    monkeypatch.setattr(
        basic_integrator.Basic,
        "prepare_wedges",
        staticmethod(
            lambda *_args, **_kwargs: (
                _single_diffraction_states(),
                wt.UInt32([0]),
                {"total": 1, "kept": 1, "threshold_pruned": 0, "roulette_pruned": 0},
                {
                    "implementation": "test_state_builder",
                    "cell_scatter_backend": "drjit_scatter_reduce",
                    "wedge_discovery_backend": "test",
                    "state_sampler": "test",
                    "point_evaluation_backend": "test",
                    "source_field_contract": "test",
                    "loop_mode": "symbolic",
                },
            )
        ),
    )
    monkeypatch.setattr(
        basic_integrator,
        "_capture_free_bytes",
        lambda **_kwargs: 64 * 1024 * 1024,
    )

    class RayDDiffractionScene:
        def __init__(self):
            self.calls = []

        def accum_dfr_direct(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([0.25]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([0]),
                keller_count=wt.Int32([7]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([7]),
            )

    scene = RayDDiffractionScene()
    grid = _filter_test_grid(nx=1, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)
    path_counts = wt.PathCounts()

    result = Basic.run_diffraction(
        collect_wedges=True,
        diff_state_store=object(),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        samples_per_tx=8,
        seed=11,
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
        ),
        scene=scene,
        grid=grid,
        diff_gain_scale=wt.Float(1.0),
        weighted_diagnostics=diagnostics,
        collect_ad_tapes=False,
        loop_mode="symbolic",
        return_timing=False,
        timing=None,
        path_counts=path_counts,
        resolved_accumulation_backend="native_monte_carlo",
    )

    assert len(scene.calls) == 1
    call = scene.calls[0]
    assert call["samples"] == 8
    assert call["direct_samples"] == 0
    assert call["keller_samples"] == 8
    assert call["sample_sequence"] == "hash"
    assert scalar(path_counts.diffraction) == 7
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.array([0.25], dtype=np.float32),
    )
    assert result.runtime_backend["implementation"] == "rayd_accum_dfr_direct"
    assert result.runtime_backend["cell_scatter_backend"] == "rayd_optix_atomic_add"


def test_basic_diffraction_rayd_optix_accepts_ad_tape_collection(monkeypatch):
    monkeypatch.setattr(
        basic_integrator.Basic,
        "prepare_wedges",
        staticmethod(
            lambda *_args, **_kwargs: (
                _single_diffraction_states(),
                wt.UInt32([0]),
                {"total": 1, "kept": 1, "threshold_pruned": 0, "roulette_pruned": 0},
                {
                    "implementation": "test_state_builder",
                    "cell_scatter_backend": "drjit_scatter_reduce",
                    "wedge_discovery_backend": "test",
                    "state_sampler": "test",
                    "point_evaluation_backend": "test",
                    "source_field_contract": "test",
                    "loop_mode": "symbolic",
                },
            )
        ),
    )
    monkeypatch.setattr(
        basic_integrator,
        "_capture_free_bytes",
        lambda **_kwargs: 64 * 1024 * 1024,
    )
    rayd_calls = []

    def fake_rayd_diffraction(**kwargs):
        rayd_calls.append(kwargs)
        kwargs["path_counts"].diffraction += wt.UInt32(8)
        return 0, 8, 0

    monkeypatch.setattr(
        basic_integrator.Basic,
        "_run_rayd_order1_diffraction",
        staticmethod(fake_rayd_diffraction),
    )

    path_counts = wt.PathCounts()
    Basic.run_diffraction(
        collect_wedges=True,
        diff_state_store=object(),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        samples_per_tx=8,
        seed=11,
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
        ),
        scene=SimpleNamespace(),
        grid=_filter_test_grid(nx=1, ny=1),
        diff_gain_scale=wt.Float(1.0),
        weighted_diagnostics=_empty_radio_map(1),
        collect_ad_tapes=True,
        loop_mode="symbolic",
        return_timing=False,
        timing=None,
        path_counts=path_counts,
        resolved_accumulation_backend="native_monte_carlo",
    )

    assert rayd_calls
    assert scalar(path_counts.diffraction) == 8


def test_basic_diffraction_rayd_optix_matches_drjit_fixed_seed_contract(monkeypatch):
    states = _single_diffraction_states()
    runtime_backend = {
        "implementation": "test_state_builder",
        "cell_scatter_backend": "drjit_scatter_reduce",
        "wedge_discovery_backend": "test",
        "state_sampler": "test",
        "point_evaluation_backend": "test",
        "source_field_contract": "test",
        "loop_mode": "symbolic",
    }

    monkeypatch.setattr(
        basic_integrator.Basic,
        "prepare_wedges",
        staticmethod(
            lambda *_args, **_kwargs: (
                states,
                wt.UInt32([0]),
                {"total": 1, "kept": 1, "threshold_pruned": 0, "roulette_pruned": 0},
                dict(runtime_backend),
            )
        ),
    )
    monkeypatch.setattr(
        basic_integrator,
        "_capture_free_bytes",
        lambda **_kwargs: 64 * 1024 * 1024,
    )

    def fake_trace_batches(**kwargs):
        kwargs["weighted_diagnostics"]["incoherent"]["diffraction"] += wt.Float([0.375])
        return wt.UInt32(5)

    monkeypatch.setattr(
        basic_integrator.Diffraction,
        "trace_batches",
        staticmethod(fake_trace_batches),
    )

    class RayDDiffractionScene:
        def accum_dfr_direct(self, **_kwargs):
            return SimpleNamespace(
                power=wt.Float([0.375]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([0]),
                keller_count=wt.Int32([5]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([5]),
            )

    def run(accumulate_primal: str):
        grid = _filter_test_grid(nx=1, ny=1)
        diagnostics = _empty_radio_map(grid.n_cells)
        path_counts = wt.PathCounts()
        result = Basic.run_diffraction(
            collect_wedges=True,
            diff_state_store=object(),
            tx_pos=wt.Point3f(0.0, 0.0, 1.0),
            samples_per_tx=5,
            seed=17,
            config=SimpleNamespace(
                wavelength=0.125,
                k=50.0,
                diffraction_execution=DiffractionExecutionConfig(
                    accumulate_primal=accumulate_primal,
                ),
            ),
            scene=RayDDiffractionScene(),
            grid=grid,
            diff_gain_scale=wt.Float(1.0),
            weighted_diagnostics=diagnostics,
            collect_ad_tapes=False,
            loop_mode="symbolic",
            return_timing=False,
            timing=None,
            path_counts=path_counts,
            resolved_accumulation_backend="native_monte_carlo",
        )
        return result, diagnostics, path_counts

    drjit_result, drjit_diagnostics, drjit_counts = run("drjit")
    rayd_result, rayd_diagnostics, rayd_counts = run("rayd_optix")

    np.testing.assert_allclose(
        np.asarray(rayd_diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.asarray(drjit_diagnostics["incoherent"]["diffraction"], dtype=np.float32),
    )
    assert scalar(rayd_counts.diffraction) == scalar(drjit_counts.diffraction)
    assert drjit_result.runtime_backend["cell_scatter_backend"] == "drjit_scatter_reduce"
    assert rayd_result.runtime_backend["cell_scatter_backend"] == "rayd_optix_atomic_add"


def test_gaussian_power_filter_preserves_constant_map():
    grid = _filter_test_grid()
    values = wt.Float([2.0] * 9)

    filtered = apply_component_filter(
        values,
        ComponentFilterConfig(method="gaussian", radius=1, sigma=1.0),
        grid=grid,
    )

    np.testing.assert_allclose(
        np.asarray(filtered, dtype=np.float32),
        np.full(9, 2.0, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_gaussian_power_filter_spreads_impulse():
    grid = _filter_test_grid()
    values = wt.Float([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    filtered = apply_component_filter(
        values,
        ComponentFilterConfig(method="gaussian", radius=1, sigma=1.0),
        grid=grid,
    )
    arr = np.asarray(filtered, dtype=np.float32)

    assert 0.0 < arr[4] < 1.0
    assert np.all(arr > 0.0)


def test_bilateral_power_filter_preserves_step_better_than_gaussian():
    grid = _filter_test_grid(nx=5, ny=1)
    values = wt.Float([0.0, 0.0, 0.0, 10.0, 10.0])

    gaussian = apply_component_filter(
        values,
        ComponentFilterConfig(method="gaussian", radius=1, sigma=1.0),
        grid=grid,
    )
    bilateral = apply_component_filter(
        values,
        ComponentFilterConfig(
            method="bilateral",
            radius=1,
            sigma=1.0,
            range_sigma=0.25,
        ),
        grid=grid,
    )

    assert np.asarray(bilateral, dtype=np.float32)[2] < np.asarray(
        gaussian,
        dtype=np.float32,
    )[2]


def test_power_filtering_changes_only_configured_components():
    grid = _filter_test_grid()
    reflection = wt.Float([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    diffraction = wt.Float([3.0] * 9)
    diagnostics = {
        "incoherent": {
            "reflection": reflection,
            "diffraction": diffraction,
        }
    }

    apply_power_filtering(
        diagnostics,
        filtering=FilterConfig(
            reflection=ComponentFilterConfig(method="gaussian", radius=1, sigma=1.0),
        ),
        grid=grid,
    )

    assert np.asarray(diagnostics["incoherent"]["reflection"], dtype=np.float32)[4] < 1.0
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.full(9, 3.0, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_disabled_power_filter_returns_input_array():
    grid = _filter_test_grid()
    values = wt.Float([1.0, 2.0, 3.0, 4.0])

    filtered = apply_component_filter(values, None, grid=grid)

    assert filtered is values


def test_power_filter_is_differentiable():
    grid = _filter_test_grid()
    values = wt.Float([0.1, 0.2, 0.3, 0.4, 1.0, 0.6, 0.7, 0.8, 0.9])
    dr.enable_grad(values)
    filtered = apply_component_filter(
        values,
        ComponentFilterConfig(method="gaussian", radius=1, sigma=1.0),
        grid=grid,
    )

    dr.set_grad(values, wt.Float([1.0] * 9))
    jvp = dr.forward_to(
        filtered,
        flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad,
    )
    arr = np.asarray(jvp, dtype=np.float32)

    assert np.all(np.isfinite(arr))
    assert float(np.sum(np.abs(arr))) > 0.0


def test_finalize_weighted_diagnostics_filters_before_total():
    grid = _filter_test_grid()
    diagnostics = _empty_radio_map(grid.n_cells)
    diagnostics["incoherent"]["los"] = wt.Float([1.0] * 9)
    diagnostics["incoherent"]["reflection"] = wt.Float(
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    )
    diagnostics["incoherent"]["diffraction"] = wt.Float([3.0] * 9)

    finalize_weighted_diagnostics(
        diagnostics,
        shadow_boundary_mode="none",
        grid=grid,
        filtering=FilterConfig(
            reflection=ComponentFilterConfig(method="gaussian", radius=1, sigma=1.0),
        ),
    )

    reflection = np.asarray(diagnostics["incoherent"]["reflection"], dtype=np.float32)
    diffraction = np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32)
    raw_total = np.asarray(diagnostics["incoherent"]["raw_total"], dtype=np.float32)
    total = np.asarray(diagnostics["incoherent"]["total"], dtype=np.float32)

    assert reflection[4] < 1.0
    np.testing.assert_allclose(diffraction, np.full(9, 3.0, dtype=np.float32))
    np.testing.assert_allclose(raw_total, 1.0 + reflection + diffraction)
    np.testing.assert_allclose(total, raw_total)


def test_diffraction_filter_keeps_shadow_boundary_transition_power_consistent():
    grid = _filter_test_grid(nx=3, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)
    diagnostics["incoherent"]["diffraction"] = wt.Float([0.0, 1.0, 0.0])
    diagnostics["incoherent"]["diffraction_incident_transition_power"] = wt.Float(
        [0.0, 1.0, 0.0]
    )
    diagnostics["incoherent"]["continued_incident_power"] = wt.Float([4.0, 4.0, 4.0])
    diagnostics["incoherent"]["incident_shadow_boundary_weight"] = wt.Float(
        [1.0, 1.0, 1.0]
    )

    finalize_weighted_diagnostics(
        diagnostics,
        shadow_boundary_mode="utd_power_smoothing",
        grid=grid,
        filtering=FilterConfig(
            diffraction=ComponentFilterConfig(
                method="gaussian",
                radius=1,
                sigma=1.0,
            ),
        ),
    )

    total = np.asarray(diagnostics["incoherent"]["total"], dtype=np.float32)
    np.testing.assert_allclose(
        total,
        np.full(3, 1.0, dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def _metadata_for_filtering(
    config: Config,
    *,
    resolved_accumulation_backend: str = "native_monte_carlo",
):
    grid = _filter_test_grid()
    return build_metadata(
        grid=grid,
        batch_plan=wt.BatchPlan(
            ray_batch_size=1,
            ray_batch_count=1,
            ray_policy="test",
            diffraction_batch_size=0,
            diffraction_batch_count=0,
            diffraction_policy="disabled",
            free_cuda_bytes=0,
            scatter_safe_batch_cap=1,
        ),
        ray_sampling_metadata={},
        runtime_reuse={
            "cache_mode": "disabled",
            "state_preparation_hits": 0,
            "state_preparation_misses": 0,
            "state_layout": "test",
        },
        solver_controls={"execution_intent": {"kind": "radio_map_incoherent"}},
        resolved_ad_mode=False,
        ad_backend="disabled",
        mc_config=config,
        samples_per_tx=1,
        seed=0,
        accepted_hit_counts=wt.PathCounts(),
        weighted_diagnostics=_empty_radio_map(grid.n_cells),
        resolved_accumulation_backend=resolved_accumulation_backend,
        reflection_runtime_backend={},
        diffraction_runtime_backend={},
        state_pool={},
        tx_power=1.0,
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold=0.0,
        solid_angle_per_ray=1.0,
        loop_mode="symbolic",
        integrator="basic",
        noise_power=0.0,
        scalar_fn=scalar,
    )


def test_metadata_reports_power_filtering_contract():
    disabled = _metadata_for_filtering(Config())
    assert disabled["monte_carlo"]["filtering"] == {"enabled": False}

    enabled = _metadata_for_filtering(
        Config(
            filtering=FilterConfig(
                diffraction=ComponentFilterConfig(
                    method="bilateral",
                    radius=2,
                    sigma=1.25,
                    range_sigma=0.5,
                )
            )
        )
    )

    filtering = enabled["monte_carlo"]["filtering"]
    assert filtering["enabled"] is True
    assert filtering["domain"] == "incoherent_power"
    assert filtering["components"]["diffraction"]["method"] == "bilateral"
    assert filtering["shadow_boundary_transition_power"] == (
        "filtered_with_diffraction_component_when_diffraction_filtering_is_enabled"
    )
    assert filtering["contract"] == "differentiable_post_accumulation_power_denoising"


def test_metadata_reports_rayd_reflection_accumulation_contract():
    metadata = _metadata_for_filtering(
        Config(
            integrator_options=IntegratorOptions(
                accumulation_backend="rayd_reflection_accumulation",
            )
        ),
        resolved_accumulation_backend="rayd_reflection_accumulation",
    )

    assert metadata["accumulation_backend"] == {
        "requested": "rayd_reflection_accumulation",
        "resolved": "rayd_reflection_accumulation",
        "cell_accumulation_mode": "rayd_optix_atomic_add",
    }


def test_ad_result_assembly_applies_power_filtering():
    grid = _filter_test_grid()
    result = BasicIntegratorAD.result_from_components(
        grid=grid,
        tx_power=1.0,
        component_power={
            "los": wt.Float([1.0] * 9),
            "reflection": wt.Float(
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "diffraction": wt.Float([0.0] * 9),
        },
        metadata={},
        tx_pos=wt.Point3f(0.0, 0.0, 0.0),
        noise_power=0.0,
        timing=None,
        shadow_boundary_mode="none",
        filtering=FilterConfig(
            reflection=ComponentFilterConfig(method="gaussian", radius=1, sigma=1.0),
        ),
    )

    result = result.squeeze_tx(0)
    reflection = np.asarray(result.incoherent["reflection"], dtype=np.float32)
    total = np.asarray(result.incoherent["total"], dtype=np.float32)

    assert reflection[1, 1] < 1.0
    np.testing.assert_allclose(np.asarray(result.path_gain, dtype=np.float32), total)


def test_bdpt_integrator_is_concrete_entrypoint():
    assert BDPT().mode == "bdpt"


def test_bdpt_zero_diffraction_does_not_request_prefix_wedges(monkeypatch):
    grid = _filter_test_grid(nx=1, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)
    captured = {}

    setup = SimpleNamespace(
        grid=grid,
        weighted_diagnostics=diagnostics,
        timing=None,
        reflection_n_rays=1,
        seed=7,
        reflection_batch_plan=SimpleNamespace(ray_batch_size=1),
        reflection_solid_angle_per_ray=1.0,
        cell_area=1.0,
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold=0.0,
        stop_threshold_linear=0.0,
        material_omega=wt.Float(0.0),
        loop_mode="symbolic",
        resolved_accumulation_backend="rayd_reflection_accumulation",
        samples_per_tx=1,
        diff_gain_scale=wt.Float(1.0),
        tx_power=1.0,
        ray_sampling_metadata={},
        batch_plan=SimpleNamespace(),
        solid_angle_per_ray=1.0,
        reflection_runtime_backend={
            "implementation": "rayd_accumulate_reflections_complex_polarized_native_ad"
        },
    )

    monkeypatch.setattr(
        bdpt_integrator.mc_custom,
        "grad_sensitive",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(Basic, "prepare_config", staticmethod(lambda **_kwargs: setup))
    monkeypatch.setattr(
        bdpt_integrator.LoS,
        "trace",
        staticmethod(
            lambda **_kwargs: SimpleNamespace(
                power=wt.Float([0.0]),
                path_count=wt.UInt32(0),
                tape=None,
            )
        ),
    )

    def fake_run_reflection(**kwargs):
        captured.update(kwargs)
        return wt.ReflectionPhaseResult(
            path_counts=wt.PathCounts(),
            path_tape_store=None,
            diff_state_store=None,
        )

    monkeypatch.setattr(Basic, "run_reflection", staticmethod(fake_run_reflection))
    monkeypatch.setattr(
        bdpt_integrator,
        "build_metadata",
        lambda **kwargs: {
            "receiver_sampling": {},
            "metric_contract": {},
            "runtime_backends": {
                "reflection": kwargs["reflection_runtime_backend"],
            },
        },
    )

    state = BDPT().integrate(
        wt.Point3f(0.0, 0.0, 1.0),
        grid,
        Config(
            max_diffraction_order=0,
            tuning=Tuning(enable_bdpt_reflection_coupled_diffraction=True),
            integrator_options=IntegratorOptions(
                integrator="bdpt",
                accumulation_backend="rayd_reflection_accumulation",
            ),
        ),
        SimpleNamespace(),
        SimpleNamespace(enable_bdpt_reflection_coupled_diffraction=True),
        {"effective": {"reflection_max_bounces": 1, "max_diffractions": 0}},
        accumulation_backend="rayd_reflection_accumulation",
        return_primal_state=True,
    )

    assert captured["collect_wedges"] is False
    assert captured["collect_wedge_prefixes"] is False
    assert (
        state.metadata["runtime_backends"]["reflection"]["implementation"]
        == "rayd_accumulate_reflections_complex_polarized_native_ad"
    )
    assert (
        state.metadata["runtime_backends"]["reflection"]["bdpt_reflection_policy"]
        == "forward_sampled_specular_only"
    )


def test_bdpt_integrator_order1_rayd_optix_metadata(monkeypatch):
    grid = _filter_test_grid(nx=1, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)
    trace_call = {}
    metadata_call = {}

    setup = SimpleNamespace(
        grid=grid,
        weighted_diagnostics=diagnostics,
        timing=None,
        reflection_n_rays=1,
        seed=19,
        reflection_batch_plan=SimpleNamespace(ray_batch_size=1),
        reflection_solid_angle_per_ray=1.0,
        cell_area=1.0,
        rr_depth=None,
        rr_prob=1.0,
        stop_threshold=0.0,
        stop_threshold_linear=0.0,
        material_omega=wt.Float(0.0),
        loop_mode="symbolic",
        resolved_accumulation_backend="native_monte_carlo",
        samples_per_tx=4,
        diff_gain_scale=wt.Float(1.0),
        tx_power=1.0,
        ray_sampling_metadata={},
        batch_plan=wt.BatchPlan(
            ray_batch_size=1,
            ray_batch_count=1,
            ray_policy="test",
            diffraction_batch_size=4,
            diffraction_batch_count=1,
            diffraction_policy="test",
            free_cuda_bytes=0,
            scatter_safe_batch_cap=4,
        ),
        solid_angle_per_ray=1.0,
        reflection_runtime_backend={
            "implementation": "cell_center_los_plus_tx_emitted_specular_reflection_single_symbolic_batch_drjit_scatter_reduce"
        },
    )

    monkeypatch.setattr(
        bdpt_integrator.mc_custom,
        "grad_sensitive",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(Basic, "prepare_config", staticmethod(lambda **_kwargs: setup))
    monkeypatch.setattr(
        bdpt_integrator.LoS,
        "trace",
        staticmethod(
            lambda **_kwargs: SimpleNamespace(
                power=wt.Float([0.0]),
                path_count=wt.UInt32(0),
                tape=None,
            )
        ),
    )
    monkeypatch.setattr(
        Basic,
        "run_reflection",
        staticmethod(
            lambda **_kwargs: wt.ReflectionPhaseResult(
                path_counts=wt.PathCounts(),
                path_tape_store=None,
                diff_state_store=object(),
            )
        ),
    )

    def fake_trace(**kwargs):
        trace_call.update(kwargs)
        strategy_counts = {
            BDPTDiffractionMIS.DIRECT_STRATEGY: 2,
            BDPTDiffractionMIS.KELLER_STRATEGY: 2,
            BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY: 0,
        }
        return BDPTDiffractionResult(
            path_count=wt.UInt32(4),
            state_count=1,
            prefix_state_count=0,
            edge_indices=wt.UInt32([0]),
            total_edge_length=1.0,
            strategy_counts=dict(strategy_counts),
            strategy_samples=dict(strategy_counts),
            order_counts={1: dict(strategy_counts)},
            order_samples={1: dict(strategy_counts)},
            runtime_backend={
                "implementation": "rayd_accum_dfr_direct",
                "cell_scatter_backend": "rayd_optix_atomic_add",
            },
        )

    monkeypatch.setattr(BDPTDiffractionMIS, "trace", staticmethod(fake_trace))

    def fake_build_metadata(**kwargs):
        metadata_call.update(kwargs)
        return {
            "receiver_sampling": {},
            "metric_contract": {},
            "path_counts": {
                "diffraction": int(scalar(kwargs["accepted_hit_counts"].diffraction))
            },
            "runtime_backends": {
                "reflection": kwargs["reflection_runtime_backend"],
                "diffraction": kwargs["diffraction_runtime_backend"],
            },
            "monte_carlo": {"state_pool": kwargs["state_pool"]},
        }

    monkeypatch.setattr(bdpt_integrator, "build_metadata", fake_build_metadata)

    mc_config = Config(
        max_diffraction_order=1,
        tuning=Tuning(
            enable_bdpt_reflection_coupled_diffraction=False,
            shadow_boundary_mode="none",
            diffraction_execution={"accumulate_primal": "rayd_optix"},
        ),
        integrator_options=IntegratorOptions(
            integrator="bdpt",
            samples_per_tx=4,
            bdpt_diffraction_sampling="hash",
            ad=False,
        ),
    )
    state = BDPT().integrate(
        wt.Point3f(0.0, 0.0, 1.0),
        grid,
        mc_config,
        SimpleNamespace(),
        mc_config.to_trace_config(),
        {"effective": {"reflection_max_bounces": 1, "max_diffractions": 1}},
        return_primal_state=True,
    )

    assert trace_call["max_depth"] == 1
    assert trace_call["prefix_store"] is None
    assert trace_call["sample_sequence"] == "hash"
    assert trace_call["config"].diffraction_execution.accumulate_primal == "rayd_optix"
    assert metadata_call["diffraction_runtime_backend"]["cell_scatter_backend"] == "rayd_optix_atomic_add"
    assert state.metadata["runtime_backends"]["diffraction"]["implementation"] == "rayd_accum_dfr_direct"
    assert state.metadata["path_counts"]["diffraction"] == 4


def test_bdpt_pure_diffraction_allocation_excludes_reflection_suffix():
    allocation = BDPTDiffractionMIS.allocate_samples(
        100,
        max_depth=2,
        include_suffix_reflection=False,
    )

    assert set(allocation) == {1, 2}
    assert sum(sum(order.values()) for order in allocation.values()) == 100
    for order_samples in allocation.values():
        assert order_samples[BDPTDiffractionMIS.DIRECT_STRATEGY] > 0
        assert order_samples[BDPTDiffractionMIS.KELLER_STRATEGY] > 0
        assert order_samples[BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY] == 0


def test_bdpt_rayd_suffix_allocation_caps_large_native_workloads():
    budget = BDPTDiffractionMIS.rayd_adaptive_budget(
        samples_per_tx=1_000_000,
        max_depth=1,
        edge_count=8192,
        grid_cell_count=256 * 256,
        reflection_max_bounces=1,
        include_suffix_reflection=True,
    )
    allocation = BDPTDiffractionMIS.allocate_samples(
        1_000_000,
        max_depth=1,
        include_suffix_reflection=True,
        suffix_sample_cap=budget.suffix_sample_cap,
    )
    order_samples = allocation[1]

    assert sum(order_samples.values()) == 1_000_000
    assert (
        order_samples[BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY]
        == budget.suffix_sample_cap
    )
    assert budget.suffix_sample_cap > BDPTDiffractionMIS.RAYD_SUFFIX_SAMPLE_CAP
    assert order_samples[BDPTDiffractionMIS.DIRECT_STRATEGY] > 400_000
    assert order_samples[BDPTDiffractionMIS.KELLER_STRATEGY] > 400_000


def test_bdpt_rayd_adaptive_budget_tracks_scene_complexity():
    small = BDPTDiffractionMIS.rayd_adaptive_budget(
        samples_per_tx=1_000_000,
        max_depth=2,
        edge_count=64,
        grid_cell_count=64 * 64,
        reflection_max_bounces=1,
        include_suffix_reflection=True,
    )
    munich_like = BDPTDiffractionMIS.rayd_adaptive_budget(
        samples_per_tx=1_000_000,
        max_depth=2,
        edge_count=8192,
        grid_cell_count=512 * 512,
        reflection_max_bounces=2,
        include_suffix_reflection=True,
    )

    assert small.policy == "adaptive_bucket_v1"
    assert munich_like.policy == "adaptive_bucket_v1"
    assert munich_like.prefix_state_sample_cap > small.prefix_state_sample_cap
    assert munich_like.suffix_sample_cap > small.suffix_sample_cap
    assert munich_like.prefix_state_sample_cap <= BDPTDiffractionMIS.RAYD_PREFIX_STATE_SAMPLE_MAX
    assert munich_like.suffix_sample_cap <= BDPTDiffractionMIS.RAYD_SUFFIX_SAMPLE_MAX


def test_bdpt_rayd_prefix_state_sampling_rescales_source_power(monkeypatch):
    captured = {}

    def fake_best_edge_indices_from_hit_data(**kwargs):
        return wt.Int32([0] * int(dr.width(kwargs["prim_index"])))

    def fake_from_edge_indices_with_sources(cls, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(edge_index=kwargs["edge_idx"])

    monkeypatch.setattr(
        DiffractionEdgeSampler,
        "best_edge_indices_from_hit_data",
        staticmethod(fake_best_edge_indices_from_hit_data),
    )
    monkeypatch.setattr(
        DiffractionStates,
        "from_edge_indices_with_sources",
        classmethod(fake_from_edge_indices_with_sources),
    )
    count = 8
    zero = wt.Float([0.0] * count)
    prefix_store = SimpleNamespace(
        count=wt.UInt32(count),
        ray_directions=wt.Vector3f(zero, zero, wt.Float([-1.0] * count)),
        prim_index=wt.Int32(list(range(count))),
        hit_p=wt.Point3f(zero, zero, zero),
        hit_n=wt.Vector3f(zero, zero, wt.Float([1.0] * count)),
        hit_geo_n=wt.Vector3f(zero, zero, wt.Float([1.0] * count)),
        source_pos=wt.Point3f(zero, zero, wt.Float([1.0] * count)),
        source_power=wt.Float([1.0] * count),
        prefix_reflection_depth=wt.Int32([1] * count),
        prefix_initial_ray_dir=wt.Vector3f(zero, zero, wt.Float([-1.0] * count)),
        prefix_prim_by_bounce=(),
    )

    states = BDPTDiffractionMIS.prepare_prefix_states(
        prefix_store=prefix_store,
        scene=SimpleNamespace(),
        config=SimpleNamespace(),
        max_states=4,
        seed=3,
    )

    assert int(dr.width(states.edge_index)) == 4
    assert int(dr.width(captured["source_power"])) == 4
    np.testing.assert_allclose(
        np.asarray(captured["source_power"], dtype=np.float32),
        np.full(4, 2.0, dtype=np.float32),
    )


def test_bdpt_rayd_prefix_state_sampling_preserves_edge_buckets(monkeypatch):
    captured = {}

    def fake_best_edge_indices_from_hit_data(**_kwargs):
        return wt.Int32([0, 0, 0, 0, 1, 1, 1, 1])

    def fake_from_edge_indices_with_sources(cls, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(edge_index=kwargs["edge_idx"])

    monkeypatch.setattr(
        DiffractionEdgeSampler,
        "best_edge_indices_from_hit_data",
        staticmethod(fake_best_edge_indices_from_hit_data),
    )
    monkeypatch.setattr(
        DiffractionStates,
        "from_edge_indices_with_sources",
        classmethod(fake_from_edge_indices_with_sources),
    )
    count = 8
    zero = wt.Float([0.0] * count)
    prefix_store = SimpleNamespace(
        count=wt.UInt32(count),
        ray_directions=wt.Vector3f(zero, zero, wt.Float([-1.0] * count)),
        prim_index=wt.Int32(list(range(count))),
        hit_p=wt.Point3f(zero, zero, zero),
        hit_n=wt.Vector3f(zero, zero, wt.Float([1.0] * count)),
        hit_geo_n=wt.Vector3f(zero, zero, wt.Float([1.0] * count)),
        source_pos=wt.Point3f(zero, zero, wt.Float([1.0] * count)),
        source_power=wt.Float([1.0] * count),
        prefix_reflection_depth=wt.Int32([1] * count),
        prefix_initial_ray_dir=wt.Vector3f(zero, zero, wt.Float([-1.0] * count)),
        prefix_prim_by_bounce=(),
    )

    states = BDPTDiffractionMIS.prepare_prefix_states(
        prefix_store=prefix_store,
        scene=SimpleNamespace(_selected_edge_runtime=lambda: {"n_edges": 2}),
        config=SimpleNamespace(),
        max_states=4,
        seed=3,
    )

    assert int(dr.width(states.edge_index)) == 4
    np.testing.assert_array_equal(
        np.sort(np.asarray(captured["edge_idx"], dtype=np.int32)),
        np.array([0, 0, 1, 1], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(captured["source_power"], dtype=np.float32),
        np.full(4, 2.0, dtype=np.float32),
    )


def test_bdpt_order1_rayd_optix_uses_native_accumulation(monkeypatch):
    def forbidden_strategy(**_kwargs):
        raise AssertionError("rayd_optix order-1 diffraction should bypass DrJit strategies")

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_direct_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_keller_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_suffix_reflection_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": _single_diffraction_states(),
                "recursive": _single_diffraction_states(),
                "prefix_state_count": 0,
            }
        ),
    )

    class RayDDiffractionScene:
        def __init__(self):
            self.calls = []

        def _selected_edge_runtime(self):
            return {"n_edges": 1}

        def accum_dfr_direct(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([0.5]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([5]),
                keller_count=wt.Int32([4]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([9]),
            )

    scene = RayDDiffractionScene()
    grid = _filter_test_grid(nx=1, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)

    result = BDPTDiffractionMIS.trace(
        scene=scene,
        grid=grid,
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            enable_bdpt_reflection_coupled_diffraction=False,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
            reflection_max_bounces=1,
        ),
        samples_per_tx=9,
        seed=5,
        diff_gain_scale=wt.Float(1.0),
        cell_area=1.0,
        weighted_diagnostics=diagnostics,
        loop_mode="symbolic",
        max_depth=1,
        sample_sequence="hash",
        prefix_store=None,
        collect_ad_tapes=False,
    )

    assert len(scene.calls) == 1
    assert scene.calls[0]["direct_samples"] == 5
    assert scene.calls[0]["keller_samples"] == 4
    assert scalar(result.path_count) == 9
    assert result.strategy_counts[BDPTDiffractionMIS.DIRECT_STRATEGY] == 5
    assert result.strategy_counts[BDPTDiffractionMIS.KELLER_STRATEGY] == 4
    assert result.strategy_samples[BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY] == 0
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )


def test_bdpt_rayd_optix_reflection_coupled_diffraction_uses_native_suffix(monkeypatch):
    def forbidden_strategy(**_kwargs):
        raise AssertionError("rayd_optix suffix diffraction should bypass DrJit strategies")

    monkeypatch.setattr(BDPTDiffractionMIS, "trace_direct_batches", staticmethod(forbidden_strategy))
    monkeypatch.setattr(BDPTDiffractionMIS, "trace_keller_batches", staticmethod(forbidden_strategy))
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_suffix_reflection_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": _single_diffraction_states(),
                "recursive": _single_diffraction_states(),
                "prefix_state_count": 0,
            }
        ),
    )

    class RayDSuffixScene:
        def __init__(self):
            self.calls = []

        def _selected_edge_runtime(self):
            return {"n_edges": 1}

        def accum_dfr_direct(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([0.25]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([3]),
                keller_count=wt.Int32([3]),
                suffix_count=wt.Int32([2]),
                vis_rejects=wt.Int32([0]),
                edge_vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([8]),
            )

    scene = RayDSuffixScene()
    diagnostics = _empty_radio_map(1)
    result = BDPTDiffractionMIS.trace(
        scene=scene,
        grid=_filter_test_grid(nx=1, ny=1),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            enable_bdpt_reflection_coupled_diffraction=True,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
            reflection_max_bounces=1,
        ),
        samples_per_tx=8,
        seed=7,
        diff_gain_scale=wt.Float(1.0),
        cell_area=1.0,
        weighted_diagnostics=diagnostics,
        loop_mode="symbolic",
        max_depth=1,
        sample_sequence="hash",
        prefix_store=object(),
        collect_ad_tapes=False,
    )

    assert len(scene.calls) == 2
    assert scene.calls[0]["direct_samples"] == 3
    assert scene.calls[0]["keller_samples"] == 3
    assert scene.calls[0]["suffix_samples"] == 0
    assert scene.calls[1]["direct_samples"] == 0
    assert scene.calls[1]["keller_samples"] == 0
    assert scene.calls[1]["suffix_samples"] == 2
    assert scalar(result.path_count) == 8
    assert result.strategy_counts[BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY] == 2
    assert result.strategy_samples[BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY] == 2
    assert result.runtime_backend["suffix_reflection"] == "rayd_optix_native"
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )


def test_bdpt_rayd_optix_accepts_ad_tape_collection(monkeypatch):
    states = _single_diffraction_states()
    calls = []

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": states,
                "direct": states,
                "prefix": None,
                "recursive": states,
                "prefix_state_count": 0,
            }
        ),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "_trace_order1_rayd_optix",
        staticmethod(
            lambda **kwargs: calls.append(kwargs)
            or BDPTDiffractionResult.zero(
                state_count=1,
                order_counts={1: BDPTDiffractionMIS._zero_strategy_counts()},
                order_samples=kwargs["strategy_samples"],
            )
        ),
    )

    BDPTDiffractionMIS.trace(
        scene=SimpleNamespace(),
        grid=_filter_test_grid(nx=1, ny=1),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            enable_bdpt_reflection_coupled_diffraction=False,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
            reflection_max_bounces=1,
        ),
        samples_per_tx=8,
        seed=7,
        diff_gain_scale=wt.Float(1.0),
        cell_area=1.0,
        weighted_diagnostics=_empty_radio_map(1),
        loop_mode="symbolic",
        max_depth=1,
        sample_sequence="hash",
        prefix_store=None,
        collect_ad_tapes=True,
    )

    assert calls


def test_bdpt_ad_accepts_reflection_coupled_suffix_with_rayd_native_ad(monkeypatch):
    states = _single_diffraction_states()
    calls = []

    class RayDADScene:
        def _selected_edge_runtime(self):
            return {"n_edges": 1}

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": states,
                "direct": states,
                "prefix": None,
                "recursive": states,
                "prefix_state_count": 0,
            }
        ),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "_trace_order1_rayd_optix",
        staticmethod(
            lambda **kwargs: calls.append(kwargs)
            or BDPTDiffractionResult.zero(
                state_count=1,
                order_counts={1: BDPTDiffractionMIS._zero_strategy_counts()},
                order_samples=kwargs["strategy_samples"],
            )
        ),
    )

    BDPTDiffractionMIS.trace(
        scene=RayDADScene(),
        grid=_filter_test_grid(nx=1, ny=1),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            enable_bdpt_reflection_coupled_diffraction=True,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
            reflection_max_bounces=1,
        ),
        samples_per_tx=8,
        seed=7,
        diff_gain_scale=wt.Float(1.0),
        cell_area=1.0,
        weighted_diagnostics=_empty_radio_map(1),
        loop_mode="symbolic",
        max_depth=1,
        sample_sequence="hash",
        prefix_store=None,
        collect_ad_tapes=True,
    )

    assert calls
    assert calls[0]["strategy_samples"][1][BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY] > 0


def test_bdpt_order2_rayd_optix_uses_native_direct_and_keller_chain(monkeypatch):
    def forbidden_strategy(**_kwargs):
        raise AssertionError("rayd_optix BDPT diffraction should bypass DrJit strategies")

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_direct_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_direct_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_keller_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_keller_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_suffix_reflection_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": _single_diffraction_states(),
                "recursive": _single_diffraction_states(),
                "prefix_state_count": 0,
            }
        ),
    )

    class RayDDiffractionScene:
        def __init__(self):
            self.order1_calls = []
            self.chain_calls = []

        def _selected_edge_runtime(self):
            return {"n_edges": 1}

        def accum_dfr_direct(self, **kwargs):
            self.order1_calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([0.5]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([kwargs["direct_samples"] + kwargs["keller_samples"]]),
            )

        def accum_dfr(self, **kwargs):
            self.chain_calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([1.25]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                edge_vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([kwargs["direct_samples"] + kwargs["keller_samples"]]),
            )

    scene = RayDDiffractionScene()
    grid = _filter_test_grid(nx=1, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)
    expected_samples = BDPTDiffractionMIS.allocate_samples(
        10,
        max_depth=2,
        include_suffix_reflection=False,
    )

    result = BDPTDiffractionMIS.trace(
        scene=scene,
        grid=grid,
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            enable_bdpt_reflection_coupled_diffraction=False,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
            reflection_max_bounces=1,
        ),
        samples_per_tx=10,
        seed=13,
        diff_gain_scale=wt.Float(1.0),
        cell_area=1.0,
        weighted_diagnostics=diagnostics,
        loop_mode="symbolic",
        max_depth=2,
        sample_sequence="hash",
        prefix_store=None,
        collect_ad_tapes=False,
    )

    assert len(scene.order1_calls) == 1
    assert len(scene.chain_calls) == 1
    order1_call = scene.order1_calls[0]
    assert order1_call["direct_samples"] == expected_samples[1][BDPTDiffractionMIS.DIRECT_STRATEGY]
    assert order1_call["keller_samples"] == expected_samples[1][BDPTDiffractionMIS.KELLER_STRATEGY]
    call = scene.chain_calls[0]
    assert call["max_order"] == 2
    assert call["samples"] == 10
    assert call["direct_samples"] == expected_samples[2][BDPTDiffractionMIS.DIRECT_STRATEGY]
    assert call["keller_samples"] == expected_samples[2][BDPTDiffractionMIS.KELLER_STRATEGY]
    assert call["sample_sequence"] == "hash"
    assert result.order_counts[2][BDPTDiffractionMIS.DIRECT_STRATEGY] == call["direct_samples"]
    assert result.order_counts[2][BDPTDiffractionMIS.KELLER_STRATEGY] == call["keller_samples"]
    assert result.runtime_backend["implementation"] == "rayd_accum_dfr_native_orders1_to_2"
    assert result.runtime_backend["native_scope"] == "orders1_to_2_direct_and_keller_no_drjit_fallback"
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.array([1.75], dtype=np.float32),
    )


def test_bdpt_order2_rayd_optix_ad_uses_native_suffix_chain(monkeypatch):
    def forbidden_strategy(**_kwargs):
        raise AssertionError("rayd_optix BDPT suffix AD should bypass DrJit strategies")

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_direct_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_direct_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_keller_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_keller_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_suffix_reflection_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": _single_diffraction_states(),
                "recursive": _single_diffraction_states(),
                "prefix_state_count": 0,
            }
        ),
    )

    class RayDDiffractionScene:
        def __init__(self):
            self.order1_calls = []
            self.chain_calls = []

        def _selected_edge_runtime(self):
            return {"n_edges": 1}

        def accum_dfr_direct(self, **kwargs):
            self.order1_calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([0.5]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([kwargs["suffix_samples"]]),
                vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([
                    kwargs["direct_samples"] + kwargs["keller_samples"] + kwargs["suffix_samples"]
                ]),
            )

        def accum_dfr(self, **kwargs):
            self.chain_calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([1.25]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([kwargs["suffix_samples"]]),
                vis_rejects=wt.Int32([0]),
                edge_vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([
                    kwargs["direct_samples"] + kwargs["keller_samples"] + kwargs["suffix_samples"]
                ]),
            )

    scene = RayDDiffractionScene()
    grid = _filter_test_grid(nx=1, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)

    result = BDPTDiffractionMIS.trace(
        scene=scene,
        grid=grid,
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            enable_bdpt_reflection_coupled_diffraction=True,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
            reflection_max_bounces=1,
        ),
        samples_per_tx=12,
        seed=13,
        diff_gain_scale=wt.Float(1.0),
        cell_area=1.0,
        weighted_diagnostics=diagnostics,
        loop_mode="symbolic",
        max_depth=2,
        sample_sequence="hash",
        prefix_store=None,
        collect_ad_tapes=True,
    )

    assert len(scene.order1_calls) == 1
    assert len(scene.chain_calls) == 1
    assert scene.order1_calls[0]["suffix_samples"] > 0
    assert scene.chain_calls[0]["max_order"] == 2
    assert scene.chain_calls[0]["suffix_samples"] > 0
    assert result.order_counts[2][BDPTDiffractionMIS.SUFFIX_REFLECTION_STRATEGY] == (
        scene.chain_calls[0]["suffix_samples"]
    )
    assert result.runtime_backend["suffix_reflection"] == "rayd_optix_native"
    assert "no_drjit_fallback" in result.runtime_backend["native_scope"]


def test_bdpt_order3_rayd_optix_uses_native_direct_and_keller_chain(monkeypatch):
    def forbidden_strategy(**_kwargs):
        raise AssertionError("rayd_optix BDPT diffraction should bypass DrJit strategies")

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_direct_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_direct_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_keller_batches",
        staticmethod(forbidden_strategy),
    )

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_keller_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_suffix_reflection_batches",
        staticmethod(forbidden_strategy),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": _single_diffraction_states(),
                "recursive": _two_recursive_diffraction_states(),
                "prefix_state_count": 0,
            }
        ),
    )

    class RayDDiffractionScene:
        def __init__(self):
            self.order1_calls = []
            self.chain_calls = []

        def _selected_edge_runtime(self):
            return {"n_edges": 3}

        def accum_dfr_direct(self, **kwargs):
            self.order1_calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([1.0]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([kwargs["direct_samples"] + kwargs["keller_samples"]]),
            )

        def accum_dfr(self, **kwargs):
            self.chain_calls.append(kwargs)
            return SimpleNamespace(
                power=wt.Float([float(kwargs["max_order"])]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                edge_vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([kwargs["direct_samples"] + kwargs["keller_samples"]]),
            )

    scene = RayDDiffractionScene()
    grid = _filter_test_grid(nx=1, ny=1)
    diagnostics = _empty_radio_map(grid.n_cells)
    expected_samples = BDPTDiffractionMIS.allocate_samples(
        12,
        max_depth=3,
        include_suffix_reflection=False,
    )

    result = BDPTDiffractionMIS.trace(
        scene=scene,
        grid=grid,
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(
            wavelength=0.125,
            k=50.0,
            enable_bdpt_reflection_coupled_diffraction=False,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="rayd_optix",
            ),
            reflection_max_bounces=1,
        ),
        samples_per_tx=12,
        seed=17,
        diff_gain_scale=wt.Float(1.0),
        cell_area=1.0,
        weighted_diagnostics=diagnostics,
        loop_mode="symbolic",
        max_depth=3,
        sample_sequence="hash",
        prefix_store=None,
        collect_ad_tapes=False,
    )

    assert len(scene.order1_calls) == 1
    assert [call["max_order"] for call in scene.chain_calls] == [2, 3]
    assert scene.order1_calls[0]["direct_samples"] == expected_samples[1][BDPTDiffractionMIS.DIRECT_STRATEGY]
    assert scene.order1_calls[0]["keller_samples"] == expected_samples[1][BDPTDiffractionMIS.KELLER_STRATEGY]
    assert scene.chain_calls[0]["direct_samples"] == expected_samples[2][BDPTDiffractionMIS.DIRECT_STRATEGY]
    assert scene.chain_calls[1]["direct_samples"] == expected_samples[3][BDPTDiffractionMIS.DIRECT_STRATEGY]
    assert scene.chain_calls[0]["keller_samples"] == expected_samples[2][BDPTDiffractionMIS.KELLER_STRATEGY]
    assert scene.chain_calls[1]["keller_samples"] == expected_samples[3][BDPTDiffractionMIS.KELLER_STRATEGY]
    assert result.order_counts[3][BDPTDiffractionMIS.DIRECT_STRATEGY] == scene.chain_calls[1]["direct_samples"]
    assert result.order_counts[3][BDPTDiffractionMIS.KELLER_STRATEGY] == scene.chain_calls[1]["keller_samples"]
    assert result.runtime_backend["implementation"] == "rayd_accum_dfr_native_orders1_to_3"
    assert result.runtime_backend["max_native_order"] == 3
    np.testing.assert_allclose(
        np.asarray(diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.array([6.0], dtype=np.float32),
    )


def test_bdpt_rayd_optix_matches_drjit_strategy_counts_and_power_contract(monkeypatch):
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "prepare_states",
        staticmethod(
            lambda **_kwargs: {
                "initial": _single_diffraction_states(),
                "recursive": _two_recursive_diffraction_states(),
                "prefix_state_count": 0,
            }
        ),
    )

    def add_power(weighted_diagnostics, value: float):
        weighted_diagnostics["incoherent"]["diffraction"] = (
            weighted_diagnostics["incoherent"]["diffraction"] + wt.Float([value])
        )

    def direct_power(order: int) -> float:
        return 0.125 * float(order)

    def keller_power(order: int) -> float:
        return 0.25 * float(order)

    def fake_trace_direct_batches(**kwargs):
        add_power(kwargs["weighted_diagnostics"], direct_power(1))
        return wt.UInt32(kwargs["direct_samples"])

    def fake_trace_keller_batches(**kwargs):
        add_power(kwargs["weighted_diagnostics"], keller_power(1))
        return wt.UInt32(kwargs["keller_samples"])

    def fake_trace_chain_direct_batches(**kwargs):
        add_power(kwargs["weighted_diagnostics"], direct_power(int(kwargs["order"])))
        return wt.UInt32(kwargs["direct_samples"])

    def fake_trace_chain_keller_batches(**kwargs):
        add_power(kwargs["weighted_diagnostics"], keller_power(int(kwargs["order"])))
        return wt.UInt32(kwargs["keller_samples"])

    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_direct_batches",
        staticmethod(fake_trace_direct_batches),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_keller_batches",
        staticmethod(fake_trace_keller_batches),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_direct_batches",
        staticmethod(fake_trace_chain_direct_batches),
    )
    monkeypatch.setattr(
        BDPTDiffractionMIS,
        "trace_chain_keller_batches",
        staticmethod(fake_trace_chain_keller_batches),
    )

    class RayDParityScene:
        def _selected_edge_runtime(self):
            return {"n_edges": 3}

        def accum_dfr_direct(self, **kwargs):
            power = 0.0
            if int(kwargs["direct_samples"]) > 0:
                power += direct_power(1)
            if int(kwargs["keller_samples"]) > 0:
                power += keller_power(1)
            return SimpleNamespace(
                power=wt.Float([power]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                edge_vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([kwargs["direct_samples"] + kwargs["keller_samples"]]),
            )

        def accum_dfr(self, **kwargs):
            order = int(kwargs["max_order"])
            power = 0.0
            if int(kwargs["direct_samples"]) > 0:
                power += direct_power(order)
            if int(kwargs["keller_samples"]) > 0:
                power += keller_power(order)
            return SimpleNamespace(
                power=wt.Float([power]),
                field_x=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_y=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                field_z=wt.Complex2f(wt.Float([0.0]), wt.Float([0.0])),
                direct_count=wt.Int32([kwargs["direct_samples"]]),
                keller_count=wt.Int32([kwargs["keller_samples"]]),
                suffix_count=wt.Int32([0]),
                vis_rejects=wt.Int32([0]),
                edge_vis_rejects=wt.Int32([0]),
                utd_rejects=wt.Int32([0]),
                edge_uses=wt.Int32([kwargs["direct_samples"] + kwargs["keller_samples"]]),
            )

    def run_mode(mode: str):
        grid = _filter_test_grid(nx=1, ny=1)
        diagnostics = _empty_radio_map(grid.n_cells)
        result = BDPTDiffractionMIS.trace(
            scene=RayDParityScene(),
            grid=grid,
            tx_pos=wt.Point3f(0.0, 0.0, 1.0),
            config=SimpleNamespace(
                wavelength=0.125,
                k=50.0,
                enable_bdpt_reflection_coupled_diffraction=False,
                diffraction_execution=DiffractionExecutionConfig(accumulate_primal=mode),
                reflection_max_bounces=1,
            ),
            samples_per_tx=12,
            seed=31,
            diff_gain_scale=wt.Float(1.0),
            cell_area=1.0,
            weighted_diagnostics=diagnostics,
            loop_mode="symbolic",
            max_depth=3,
            sample_sequence="hash",
            prefix_store=None,
            collect_ad_tapes=False,
        )
        return result, diagnostics

    drjit_result, drjit_diagnostics = run_mode("drjit")
    rayd_result, rayd_diagnostics = run_mode("rayd_optix")

    assert scalar(rayd_result.path_count) == scalar(drjit_result.path_count)
    assert rayd_result.strategy_counts == drjit_result.strategy_counts
    assert rayd_result.order_counts == drjit_result.order_counts
    np.testing.assert_allclose(
        np.asarray(rayd_diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        np.asarray(drjit_diagnostics["incoherent"]["diffraction"], dtype=np.float32),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_diffraction_edge_sampler_custom_weight_preserves_edge_measure():
    line_length = wt.Float([2.0, 4.0])
    sample_weight = wt.Float([1.0, 7.0])
    sampler = DiffractionEdgeSampler.from_sample_weight(
        line_length=line_length,
        sample_weight=sample_weight,
    )

    assert sampler is not None
    assert sampler.total_length_scalar == pytest.approx(6.0)
    assert sampler.total_sampling_weight_scalar == pytest.approx(8.0)

    slots = sampler.sample_slots_from_uniform(wt.Float([0.05, 0.5]))
    assert int(scalar(dr.gather(wt.UInt32, slots, wt.UInt32(0)))) == 0
    assert int(scalar(dr.gather(wt.UInt32, slots, wt.UInt32(1)))) == 1

    state_slot = wt.UInt32([0, 1])
    gathered_length = dr.gather(wt.Float, line_length, state_slot)
    edge_measure_weight = sampler.edge_measure_weight(state_slot, gathered_length)
    assert float(scalar(dr.gather(wt.Float, edge_measure_weight, wt.UInt32(0)))) == pytest.approx(16.0)
    assert float(scalar(dr.gather(wt.Float, edge_measure_weight, wt.UInt32(1)))) == pytest.approx(32.0 / 7.0)
