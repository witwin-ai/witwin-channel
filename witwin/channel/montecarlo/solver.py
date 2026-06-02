"""Public solve entry for the Monte Carlo radiomap package."""

from __future__ import annotations

from dataclasses import replace

from witwin.channel.core.scene import ReceiverGrid, Scene, Transmitter
from witwin.channel.core.results import RadioMapResult, stack_radiomap_results
from witwin.channel.core.runtime import assert_scene_materials_complete
from witwin.channel.core.geometry.mesh_buffers import to_point3f
from witwin.channel.core.numerics.tensors import to_mapping_proxy
from witwin.channel.montecarlo import types as wt

from .config import Config, ResolvedTraceConfig, resolve_solver_controls
from .integrators import BDPT, Basic, Integrator


_INTEGRATORS: dict[str, type[Integrator]] = {"basic": Basic, "bdpt": BDPT}


def solve(
    *,
    scene: Scene,
    transmitter: str | Transmitter | list[str | Transmitter] | tuple[str | Transmitter, ...],
    receiver,
    config: Config | None = None,
) -> RadioMapResult:
    """Compute a Monte Carlo radiomap using ``scene.frequency``."""
    assert_scene_materials_complete(scene)
    mc_config = Config() if config is None else config
    frequency = scene.frequency
    if frequency is None:
        raise ValueError("montecarlo.solve requires Scene.frequency.")
    if isinstance(transmitter, (list, tuple)):
        if not transmitter:
            raise ValueError("montecarlo.solve transmitter list must not be empty.")
        results = tuple(
            solve(scene=scene, transmitter=item, receiver=receiver, config=mc_config)
            for item in transmitter
        )
        return stack_radiomap_results(results, noise_power=results[0].noise_power)
    integrator_options = mc_config.integrator_options
    tx_power = 1.0
    tx_polarization: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rx_polarization: tuple[float, float, float] | None = None

    tx_endpoint = scene.transmitter(transmitter)
    tx_pos = tx_endpoint.position
    tx_power = float(tx_endpoint.power)
    tx_polarization = tx_endpoint.polarization or (1.0, 0.0, 0.0)
    tx_pos = to_point3f(tx_pos, role="transmitter")

    receiver_endpoint = scene.receiver(receiver)
    if not isinstance(receiver_endpoint, ReceiverGrid):
        raise TypeError("montecarlo.solve receiver endpoint must be a ReceiverGrid.")
    grid = receiver_endpoint
    rx_polarization = receiver_endpoint.polarization

    integrator_cls = _INTEGRATORS.get(integrator_options.integrator)
    if integrator_cls is None:
        raise ValueError(f"Unsupported integrator: {integrator_options.integrator!r}.")

    trace_config = mc_config.to_trace_config()
    resolved_trace = replace(
        ResolvedTraceConfig.from_config(frequency=frequency, config=trace_config),
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
    )
    solver_controls = resolve_solver_controls(
        trace_config,
        execution_intent="radio_map_incoherent",
        max_diffractions_override=int(mc_config.max_diffraction_order),
    )
    scene.diffraction_edge_count(edge_policy=mc_config.edge_policy)
    result = integrator_cls().integrate(
        tx_pos, grid, mc_config, scene, resolved_trace, solver_controls,
        accumulation_backend=str(integrator_options.accumulation_backend),
        return_timing=True,
        tx_power=tx_power,
    )
    if not isinstance(result, RadioMapResult):
        return result
    metadata = dict(result.metadata)
    metadata["array_model"] = {
        "synthetic_array": bool(mc_config.synthetic_array),
        "assumption": "synthetic_array=True traces endpoint centers and assumes far-field plane-wave phase across the aperture; use explicit mode for near-field or large apertures.",
    }
    return replace(result, metadata=to_mapping_proxy(metadata))


__all__ = ["solve"]
