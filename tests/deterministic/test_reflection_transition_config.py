from __future__ import annotations

from types import SimpleNamespace

import pytest

from witwin.channel.deterministic.config import Config, SolveSpec, Tuning, resolve_trace_config
from witwin.channel.deterministic.reflection.detail import (
    TRACE_DETAIL_KIND,
    build_trace_detail,
    coerce_trace_detail,
)
from witwin.channel.deterministic.solver import _build_metadata


def test_reflection_transition_defaults_to_hard_mode():
    tuning = Tuning()

    assert tuning.reflection_transition_mode == "hard"
    assert tuning.reflection_f_weight_boundary_radius_wavelengths == 2.0
    assert tuning.reflection_f_weight_max_edges_per_slot == 1
    assert tuning.reflection_secondary_visibility_mode == "hard"

    resolved = resolve_trace_config(frequency=1.0e9, config=Config(tuning=tuning))

    assert resolved.reflection_transition_mode == "hard"
    assert resolved.reflection_f_weight_boundary_radius_wavelengths == 2.0
    assert resolved.reflection_f_weight_max_edges_per_slot == 1
    assert resolved.reflection_secondary_visibility_mode == "hard"


def test_reflection_transition_accepts_reference_and_native_modes():
    reference = resolve_trace_config(
        frequency=1.0e9,
        config=Config(tuning=Tuning(reflection_transition_mode="f_weight_reference")),
    )
    native = resolve_trace_config(
        frequency=1.0e9,
        config=Config(tuning=Tuning(reflection_transition_mode="f_weight_native")),
    )

    assert reference.reflection_transition_mode == "f_weight_reference"
    assert native.reflection_transition_mode == "f_weight_native"


def test_reflection_transition_rejects_invalid_mode():
    with pytest.raises(ValueError, match="reflection_transition_mode"):
        Tuning(reflection_transition_mode="smooth")


def test_reflection_secondary_visibility_accepts_f_weight_mode():
    resolved = resolve_trace_config(
        frequency=1.0e9,
        config=Config(tuning=Tuning(reflection_secondary_visibility_mode="f_weight")),
    )

    assert resolved.reflection_secondary_visibility_mode == "f_weight"


def test_reflection_secondary_visibility_rejects_invalid_mode():
    with pytest.raises(ValueError, match="reflection_secondary_visibility_mode"):
        Tuning(reflection_secondary_visibility_mode="smooth")


def test_reflection_transition_rejects_invalid_performance_tuning():
    with pytest.raises(ValueError, match="reflection_f_weight_boundary_radius_wavelengths"):
        Tuning(reflection_f_weight_boundary_radius_wavelengths=0.0)

    with pytest.raises(ValueError, match="reflection_f_weight_max_edges_per_slot"):
        Tuning(reflection_f_weight_max_edges_per_slot=0)


def test_reflection_transition_rejects_shadow_boundary_double_application():
    config = Config(
        shadow_boundary_correction=True,
        tuning=Tuning(reflection_transition_mode="f_weight_reference"),
    )

    with pytest.raises(ValueError, match="shadow_boundary_correction"):
        resolve_trace_config(frequency=1.0e9, config=config)


def test_reflection_secondary_visibility_rejects_shadow_boundary_double_application():
    config = Config(
        shadow_boundary_correction=True,
        tuning=Tuning(reflection_secondary_visibility_mode="f_weight"),
    )

    with pytest.raises(ValueError, match="shadow_boundary_correction"):
        resolve_trace_config(frequency=1.0e9, config=config)


def test_solver_metadata_records_reflection_transition_settings():
    config = Config(
        tuning=Tuning(
            reflection_transition_mode="f_weight_reference",
            reflection_f_weight_boundary_radius_wavelengths=1.5,
            reflection_f_weight_max_edges_per_slot=2,
        )
    )
    resolved = resolve_trace_config(frequency=1.0e9, config=config)
    scene = SimpleNamespace(
        tri_data=None,
        structures=(),
        device="cpu",
        diffraction_edge_count=lambda *, edge_policy: 0,
    )
    spec = SolveSpec.from_public(
        grid=SimpleNamespace(axis="z", position=1.0, bounds=((-1, 1), (-1, 1)), grid_shape=(2, 2), cell_size=None),
        config=config,
    )
    grid = SimpleNamespace(surface_mode="axis_aligned", grid_shape=(2, 2), cell_size=(1.0, 1.0))

    metadata = _build_metadata(
        scene=scene,
        resolved=resolved,
        rm_config=config,
        spec=spec,
        grid=grid,
        solver_controls={},
        timing={},
        diffraction_metadata={},
        shadow_boundary_payload=None,
    )

    assert metadata["runtime_backends"]["reflection_transition"] == {
        "mode": "f_weight_reference",
        "resolved_backend": "reference_pair_replay",
        "boundary_radius_wavelengths": 1.5,
        "max_edges_per_slot": 2,
    }
    assert metadata["runtime_backends"]["reflection_secondary_visibility"] == {
        "mode": "hard",
        "resolved_backend": "hard",
    }


def test_solver_metadata_records_secondary_visibility_settings():
    config = Config(tuning=Tuning(reflection_secondary_visibility_mode="f_weight"))
    resolved = resolve_trace_config(frequency=1.0e9, config=config)
    scene = SimpleNamespace(
        tri_data=None,
        structures=(),
        device="cpu",
        diffraction_edge_count=lambda *, edge_policy: 0,
    )
    spec = SolveSpec.from_public(
        grid=SimpleNamespace(axis="z", position=1.0, bounds=((-1, 1), (-1, 1)), grid_shape=(2, 2), cell_size=None),
        config=config,
    )
    grid = SimpleNamespace(surface_mode="axis_aligned", grid_shape=(2, 2), cell_size=(1.0, 1.0))

    metadata = _build_metadata(
        scene=scene,
        resolved=resolved,
        rm_config=config,
        spec=spec,
        grid=grid,
        solver_controls={},
        timing={},
        diffraction_metadata={},
        shadow_boundary_payload=None,
    )

    assert metadata["runtime_backends"]["reflection_secondary_visibility"] == {
        "mode": "f_weight",
        "resolved_backend": "reference_segment_f_weight",
    }


def test_solver_metadata_records_native_cuda_transition_backend():
    config = Config(tuning=Tuning(reflection_transition_mode="f_weight_native"))
    resolved = resolve_trace_config(frequency=1.0e9, config=config)
    scene = SimpleNamespace(
        tri_data=None,
        structures=(),
        device="cpu",
        diffraction_edge_count=lambda *, edge_policy: 0,
    )
    spec = SolveSpec.from_public(
        grid=SimpleNamespace(axis="z", position=1.0, bounds=((-1, 1), (-1, 1)), grid_shape=(2, 2), cell_size=None),
        config=config,
    )
    grid = SimpleNamespace(surface_mode="axis_aligned", grid_shape=(2, 2), cell_size=(1.0, 1.0))

    metadata = _build_metadata(
        scene=scene,
        resolved=resolved,
        rm_config=config,
        spec=spec,
        grid=grid,
        solver_controls={},
        timing={},
        diffraction_metadata={},
        shadow_boundary_payload=None,
    )

    assert metadata["runtime_backends"]["reflection_transition"]["resolved_backend"] == "native_cuda_f_weight"


def test_reflection_trace_detail_carries_transition_settings():
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="test",
        reflection_gain=1.0,
        reflection_transition_mode="f_weight_native",
        reflection_f_weight_boundary_radius_wavelengths=1.25,
        reflection_f_weight_max_edges_per_slot=3,
    )

    assert detail.reflection_transition_mode == "f_weight_native"
    assert detail.reflection_f_weight_boundary_radius_wavelengths == 1.25
    assert detail.reflection_f_weight_max_edges_per_slot == 3
    assert detail.reflection_secondary_visibility_mode == "hard"


def test_reflection_trace_detail_carries_secondary_visibility_settings():
    detail = build_trace_detail(
        reflection_model="materialized",
        reflection_model_source="test",
        reflection_gain=1.0,
        reflection_secondary_visibility_mode="f_weight",
    )

    assert detail.reflection_secondary_visibility_mode == "f_weight"


def test_reflection_trace_detail_payload_defaults_to_hard_transition():
    detail = coerce_trace_detail(
        {
            "detail_kind": TRACE_DETAIL_KIND,
            "reflection_model": "materialized",
            "reflection_model_source": "legacy",
            "reflection_gain": 1.0,
            "source_paths_per_bounce": (),
        }
    )

    assert detail.reflection_transition_mode == "hard"
    assert detail.reflection_f_weight_boundary_radius_wavelengths == 2.0
    assert detail.reflection_f_weight_max_edges_per_slot == 1
    assert detail.reflection_secondary_visibility_mode == "hard"
