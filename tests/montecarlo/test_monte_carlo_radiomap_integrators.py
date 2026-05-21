"""Non-GPU contract checks for standalone Monte Carlo radiomap integrators."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import drjit as dr
import numpy as np

from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.grid import Grid, GridSpec
from witwin.channel.montecarlo import ComponentFilterConfig, Config, FilterConfig, IntegratorOptions, Tuning
from witwin.channel.montecarlo import types as wt
from witwin.channel.montecarlo.integrators import (
    BDPT,
    Basic,
    Integrator,
)
from witwin.channel.montecarlo.integrators import basic as basic_integrator
from witwin.channel.montecarlo.integrators.bdpt_diffraction import BDPTDiffractionMIS
from witwin.channel.montecarlo.filtering import apply_component_filter, apply_power_filtering
from witwin.channel.montecarlo.integrators.basic import _empty_radio_map, finalize_weighted_diagnostics
from witwin.channel.montecarlo.integrators.basic_ad import BasicIntegratorAD
from witwin.channel.montecarlo.integrators.metadata import build_metadata
from witwin.channel.montecarlo.grid_ops import GridContributionStore
from witwin.channel.montecarlo.trace.diffraction import Diffraction, DiffractionEdgeSampler


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

        def trace_reflections_accumulating(self, **kwargs):
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


def test_basic_reflection_rayd_accumulation_rejects_ad_and_prefix_wedges():
    common = dict(
        scene=object(),
        grid=_filter_test_grid(nx=1, ny=1),
        tx_pos=wt.Point3f(0.0, 0.0, 1.0),
        config=SimpleNamespace(wavelength=0.125, k=50.0),
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
        weighted_diagnostics=_empty_radio_map(1),
        collect_wedges=False,
        loop_mode="symbolic",
        resolved_accumulation_backend="rayd_reflection_accumulation",
    )

    with pytest.raises(RuntimeError, match="does not support AD"):
        Basic.run_reflection(**common, collect_ad_tapes=True)

    with pytest.raises(RuntimeError, match="prefix wedge"):
        Basic.run_reflection(
            **common,
            collect_ad_tapes=False,
            collect_wedge_prefixes=True,
        )


def test_basic_integrator_rejects_rayd_accumulation_when_ad_requested():
    class SceneWithoutGrad:
        def _triangle_runtime(self):
            return None

    config = Config(
        integrator_options=IntegratorOptions(
            ad=True,
            accumulation_backend="rayd_reflection_accumulation",
        )
    )

    with pytest.raises(RuntimeError, match="does not support AD"):
        Basic().integrate(
            wt.Point3f(0.0, 0.0, 1.0),
            _filter_test_grid(nx=1, ny=1),
            config,
            SceneWithoutGrad(),
            SimpleNamespace(),
            {"effective": {"reflection_max_bounces": 1, "max_diffractions": 0}},
            accumulation_backend="rayd_reflection_accumulation",
        )


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
