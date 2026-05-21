import pytest
import witwin.channel as wt

from witwin.channel.core.physics.materials import resolve_surface_material
from witwin.channel.deterministic import Config as DeterministicConfig
from witwin.channel.deterministic.config import resolve_trace_config as resolve_deterministic_trace_config
from witwin.channel.montecarlo import Config as MonteCarloConfig
from witwin.channel.montecarlo.config import ResolvedTraceConfig as MonteCarloResolvedTraceConfig
from witwin.channel.montecarlo.trace.diffraction import DiffractionStates
from witwin.channel.montecarlo.trace.ad_support import SceneQuery
from witwin.channel.deterministic.reflection.detail import (
    build_trace_detail,
    coerce_material_context,
)
from witwin.channel.core.runtime import Material


def test_shared_surface_material_resolution_requires_scene_table():
    with pytest.raises(RuntimeError, match="scene material table"):
        resolve_surface_material(
            scene=None,
            prim_idx=wt.Int32([0]),
            default_gain=1.0,
        )


def test_monte_carlo_edge_faces_requires_scene_table():
    with pytest.raises(RuntimeError, match="scene material table"):
        DiffractionStates._face_materials(
            scene=None,
            adjacent_face0=wt.Int32([0]),
            adjacent_face1=wt.Int32([1]),
        )


def test_monte_carlo_ad_material_resolution_requires_scene_table():
    class EmptyScene:
        def _triangle_runtime(self):
            return None

    with pytest.raises(RuntimeError, match="scene material table"):
        SceneQuery.material(
            wt.Int32([0]),
            scene=EmptyScene(),
            gain=1.0,
        )


def test_resolved_trace_configs_do_not_carry_material_defaults():
    deterministic = resolve_deterministic_trace_config(
        frequency=1.0e9,
        config=DeterministicConfig(),
    )
    monte_carlo = MonteCarloResolvedTraceConfig.from_config(
        frequency=1.0e9,
        config=MonteCarloConfig().to_trace_config(),
    )

    for resolved in (deterministic, monte_carlo):
        assert not hasattr(resolved, "reflection_relative_permittivity")
        assert not hasattr(resolved, "reflection_conductivity")
        assert not hasattr(resolved, "reflection_material")
        assert not hasattr(resolved, "diffraction_material")


def test_deterministic_reflection_detail_does_not_carry_material_defaults():
    trace_detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="test",
        reflection_gain=1.0,
        source_paths_per_bounce=(),
    )
    material_context = coerce_material_context(trace_detail, default_gain=1.0)
    runtime_material = Material(reflection_coef=1.0)

    assert not hasattr(trace_detail, "default_eta_r")
    assert not hasattr(trace_detail, "default_sigma")
    assert not hasattr(material_context, "default_eta_r")
    assert not hasattr(material_context, "default_sigma")
    assert not hasattr(runtime_material, "relative_permittivity")
    assert not hasattr(runtime_material, "conductivity")
