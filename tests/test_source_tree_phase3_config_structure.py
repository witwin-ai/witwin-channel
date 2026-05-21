from __future__ import annotations

from witwin.channel._config.base import CommonTraceTuning, ResolvedTraceBase
from witwin.channel.deterministic import config as det_config
from witwin.channel.montecarlo import config as mc_config


def test_solver_tuning_classes_share_common_trace_tuning_base():
    assert issubclass(det_config.Tuning, CommonTraceTuning)
    assert issubclass(mc_config.Tuning, CommonTraceTuning)

    det_tuning = det_config.Tuning(memory_profile="memory_safe")
    mc_tuning = mc_config.Tuning(memory_profile="memory_safe")

    assert det_tuning.memory_profile == "memory_safe"
    assert mc_tuning.memory_profile == "memory_safe"
    assert det_tuning.resolution_wavelength == mc_tuning.resolution_wavelength == 0.125


def test_resolved_trace_configs_share_wave_and_budget_fields():
    assert issubclass(det_config.ResolvedTraceConfig, ResolvedTraceBase)
    assert issubclass(mc_config.ResolvedTraceConfig, ResolvedTraceBase)

    det_resolved = det_config.resolve_trace_config(
        frequency=3.0e9,
        config=det_config.Config(num_samples=7, max_bounces=1, max_diffraction_order=0),
    )
    mc_resolved = mc_config.ResolvedTraceConfig.from_config(
        frequency=3.0e9,
        config=mc_config.Config(num_samples=7, max_bounces=1, max_diffraction_order=0).to_trace_config(),
    )

    assert det_resolved.frequency == mc_resolved.frequency == 3.0e9
    assert det_resolved.reflection_n_rays == mc_resolved.reflection_n_rays == 7
    assert det_resolved.reflection_max_bounces == mc_resolved.reflection_max_bounces == 1
    assert det_resolved.max_diffractions == mc_resolved.max_diffractions == 0
    assert det_resolved.cell_size == mc_resolved.cell_size


def test_shared_solver_guardrails_keep_solver_specific_execution_intents():
    det_controls = det_config.resolve_solver_controls(
        det_config.Config(
            num_samples=4096,
            max_bounces=3,
            max_diffraction_order=2,
            tuning=det_config.Tuning(solver_mode="fast_approximate"),
        ),
        execution_intent="coherent",
    )
    mc_controls = mc_config.resolve_solver_controls(
        mc_config.Config(
            num_samples=4096,
            max_bounces=3,
            max_diffraction_order=2,
            integrator_options=mc_config.IntegratorOptions(integrator="bdpt"),
            tuning=mc_config.Tuning(solver_mode="fast_approximate"),
        ).to_trace_config(),
        execution_intent="radio_map_incoherent",
    )

    assert det_controls["effective"]["reflection_n_rays"] == 1024
    assert mc_controls["effective"]["reflection_n_rays"] == 1024
    assert det_controls["effective"]["reflection_max_bounces"] == 1
    assert mc_controls["effective"]["reflection_max_bounces"] == 1
    assert det_controls["execution_intent"]["kind"] == "coherent"
    assert mc_controls["execution_intent"]["kind"] == "radio_map_incoherent"
