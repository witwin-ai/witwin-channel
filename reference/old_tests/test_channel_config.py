"""Tests for system-level channel configuration."""

import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from witwin.channel import ChannelConfig, DiffractionExecutionConfig, TraceConfig
from witwin.channel.config import coerce_diffraction_execution
def test_channel_config_from_dict():
    payload = {
        "trace": {
            "reflection_n_rays": 128,
            "reflection_max_bounces": 3,
            "min_ray_contribution_threshold": 0.15,
            "solver_mode": "fast_approximate",
            "memory_profile": "memory_safe",
            "reflection_field_backend": "drjit",
            "tx_polarization": [0.0, 1.0, 0.0],
            "diffraction_execution": {
                "accumulate_primal": "drjit",
                "accumulate_jvp": "drjit_replay",
                "accumulate_backward": "drjit_replay",
                "suffix_backend": "drjit",
                "suffix_dda": "symbolic",
                "suffix_russian_roulette": True,
            },
        },
    }
    from_dict = ChannelConfig.from_dict(payload)
    assert from_dict.trace.reflection_n_rays == 128
    assert from_dict.trace.reflection_max_bounces == 3
    assert from_dict.trace.min_ray_contribution_threshold == 0.15
    assert from_dict.trace.solver_mode == "fast_approximate"
    assert from_dict.trace.memory_profile == "memory_safe"
    assert from_dict.trace.reflection_field_backend == "drjit"
    assert from_dict.trace.tx_polarization == (0.0, 1.0, 0.0)
    assert from_dict.trace.diffraction_execution.to_dict() == {
        "accumulate_primal": "drjit",
        "accumulate_jvp": "drjit_replay",
        "accumulate_backward": "drjit_replay",
        "suffix_backend": "drjit",
        "suffix_dda": "symbolic",
        "suffix_russian_roulette": True,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"trace": {"solver_mode": "bad_mode"}},
        {"trace": {"memory_profile": "bad_mode"}},
        {"trace": {"min_ray_contribution_threshold": -0.1}},
        {"trace": {"min_ray_contribution_threshold": 1.0}},
        {"trace": {"reflection_field_backend": "bad_mode"}},
        {"trace": {"diffraction_execution": {"accumulate_primal": "bad_mode"}}},
        {"trace": {"diffraction_execution": {"accumulate_jvp": "bad_mode"}}},
        {"trace": {"diffraction_execution": {"accumulate_backward": "bad_mode"}}},
        {"trace": {"diffraction_execution": {"suffix_backend": "bad_mode"}}},
        {"trace": {"diffraction_execution": {"suffix_dda": "bad_mode"}}},
        {"trace": {"diffraction_execution": {"suffix_dda": "evaluated"}}},
    ],
)
def test_channel_config_rejects_invalid_enums(payload):
    with pytest.raises(ValueError):
        ChannelConfig.from_dict(payload)


def test_default_execution_matches_coerced_default():
    execution = DiffractionExecutionConfig.default()
    resolved = coerce_diffraction_execution(execution)
    assert resolved == execution
    assert execution.suffix_backend == "native"


def test_trace_config_defaults_to_native_backends():
    trace = TraceConfig()
    assert trace.memory_profile == "default"
    assert trace.reflection_field_backend == "native"
    assert trace.diffraction_execution.suffix_backend == "native"


@pytest.mark.gpu
def test_scene_and_tracer_explicit_arguments_override_trace_config():
    from witwin.channel import Scene, Tracer
    config = ChannelConfig(
        trace=TraceConfig(
            reflection_n_rays=64,
            reflection_max_bounces=1,
            solver_mode="accuracy",
            reflection_field_backend="native",
            max_diffractions=1,
            diffraction_execution=DiffractionExecutionConfig(
                accumulate_primal="drjit",
                accumulate_jvp="drjit_replay",
                accumulate_backward="drjit_replay",
                suffix_backend="drjit",
                suffix_dda="symbolic",
            ),
        ),
    )

    scene = Scene(config=config)
    tracer = Tracer(
        frequency=1e9,
        scene=scene,
        config=config,
        reflection_n_rays=256,
        solver_mode="fast_approximate",
    )

    assert tracer.config.trace.reflection_n_rays == 256
    assert tracer.config.trace.solver_mode == "fast_approximate"
    assert tracer.config.trace.reflection_field_backend == "native"
    assert tracer.config.trace.diffraction_execution.suffix_backend == "drjit"
    assert tracer.config.trace.diffraction_execution.suffix_dda == "symbolic"
