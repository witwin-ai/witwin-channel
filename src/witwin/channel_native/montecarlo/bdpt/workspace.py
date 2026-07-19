"""Endpoint subpath assembly and per-solve workspace contracts for BDPT.

Split out of :mod:`montecarlo.bdpt.pipeline` to keep it within the maintenance
size budget. This module owns the two per-solve dataclasses
(``_SolvePrep`` / ``_EndpointWorkspace``), the accumulate dispatcher, and the
endpoint-subpath construction that feeds the connection-sample collectors. It
depends only on the solver's kernel facades and shared core helpers; the
enumerated discrete builders and the connection-sample collectors stay in
``pipeline`` (ADR-008 keeps the enumerated dependency owned there).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from witwin.channel_native.core.field_state import (
    receiver_polarizations,
    transmitter_polarizations,
)
from witwin.channel_native.core.kernels.metadata import AdLaunchLedger
from witwin.channel_native.scene.models import ReceiverGrid, Scene
from witwin.channel_native.montecarlo.bdpt.autograd import (
    bdpt_endpoint_connection_samples_ad,
)
from witwin.channel_native.montecarlo.bdpt.autograd_accumulate import (
    bdpt_accumulate_connection_samples_ad,
)
from witwin.channel_native.montecarlo.bdpt.kernels.paths import (
    bdpt_accumulate_connection_samples,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_subpath_state,
)

from .config import Config
from .endpoints import receiver_positions
from .sampling import make_launch_state


@dataclass(frozen=True, slots=True)
class _SolvePrep:
    """Workspace-sizing and native-capability results for one solve()."""

    native_samples: int
    native_max_depth: int
    selected_accumulation: str
    workspace_bytes: int
    info: dict[str, object]
    rayd: Any
    reflection_available: bool
    diffraction_available: bool
    transmission_available: bool


@dataclass(frozen=True, slots=True)
class _EndpointWorkspace:
    """Endpoint subpath tensors and derived counts shared across stages."""

    tx_reference: torch.Tensor
    tx_power: torch.Tensor
    rx_positions: torch.Tensor
    topology_scene: Scene
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    launch_state: dict[str, torch.Tensor]
    endpoint_subpaths: dict[str, Any]
    los_light_state: dict[str, torch.Tensor] | None
    endpoint_connection_samples: dict[str, torch.Tensor] | None
    endpoint_accumulation: dict[str, torch.Tensor] | None
    launch_count: int
    tx_count: int
    rx_count: int
    endpoint_only: bool


def _accumulate_connection_samples(
    config: Config,
    samples: dict[str, torch.Tensor],
    *,
    tx_count: int,
    rx_count: int,
    accumulation_strategy: str,
    combine_domain: str = "power",
    coeff_real: torch.Tensor | None = None,
    coeff_imag: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Accumulate dispatcher: the differentiable twin under ad_mode != 'none',
    else the bitwise primal. Both domains (power/coherent) route through the
    ADR-022 accumulate companions when AD is active (spec 6.4)."""

    if config.ad_mode != "none":
        return bdpt_accumulate_connection_samples_ad(
            samples,
            tx_count=tx_count,
            rx_count=rx_count,
            accumulation_strategy=accumulation_strategy,
            combine_domain=combine_domain,
            coeff_real=coeff_real,
            coeff_imag=coeff_imag,
        )
    return bdpt_accumulate_connection_samples(
        samples,
        tx_count=tx_count,
        rx_count=rx_count,
        accumulation_strategy=accumulation_strategy,
        combine_domain=combine_domain,
        coeff_real=coeff_real,
        coeff_imag=coeff_imag,
    )


def _reduced_light_endpoint_state(
    tx_reference: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One light endpoint per transmitter for the deterministic LoS term.

    All depth-0 light samples share the transmitter position, so N samples
    per tx are N identical rows weighted 1/N. Connecting the T unique
    endpoints with samples_per_tx=1 yields the identical estimate while the
    connection table shrinks from T*N*R rows to T*R (audit P-1/P-5).
    """

    device = tx_reference.device
    tx_count = int(tx_reference.shape[0])
    tx_ids = torch.arange(tx_count, device=device, dtype=torch.int32)
    seeds = torch.zeros((tx_count,), device=device, dtype=torch.int64)
    return bdpt_endpoint_subpath_state(
        tx_reference,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        tx_ids,
        seeds,
    )["light"]


def _live_tx_power(scene: Scene, *, reference: torch.Tensor) -> torch.Tensor:
    """Reattach the live per-tx power leaves' gradient onto the native power pack.

    ADR-022 tx_power threading: ``endpoints.transmitter_tensors`` reads
    ``float(power_w)`` and detaches the leaf. Under ad we pack the same values
    from the ``Transmitter.power_w`` tensors and add them (minus their detached
    selves) to the native ``reference`` so the returned tensor is bitwise-equal
    to the detached pack while carrying the leaves' gradient. A float ``power_w``
    packs a plain constant, so a materials-only ad graph is unchanged. The
    native endpoint kernels read the data pointer and detach on output, so the
    live gradient reaches the differentiable inputs (endpoint-connection
    companion, scattering-NEE source power) rather than the frozen subpaths."""

    powers = []
    for transmitter in scene.transmitters:
        power = transmitter.power_w
        if isinstance(power, torch.Tensor):
            powers.append(power.to(device=reference.device, dtype=reference.dtype))
        else:
            powers.append(
                torch.tensor(
                    float(power), device=reference.device, dtype=reference.dtype
                )
            )
    packed = torch.stack(powers)
    return reference + (packed - packed.detach())


def _build_endpoint_subpaths(
    scene: Scene,
    config: Config,
    *,
    grid: ReceiverGrid | None,
    transmitter_tensors_fn: Callable[[Scene], tuple[torch.Tensor, torch.Tensor]],
    selected_accumulation: str,
    ledger: AdLaunchLedger | None = None,
) -> _EndpointWorkspace:
    tx_reference, tx_power = transmitter_tensors_fn(scene)
    if config.ad_mode != "none":
        tx_power = _live_tx_power(scene, reference=tx_power)
    rx_positions = receiver_positions(scene, reference=tx_reference, grid=grid)
    topology_scene = (
        scene
        if grid is None
        else Scene(
            structures=scene.structures,
            transmitters=scene.transmitters,
            receivers=[grid],
            frequency=scene.frequency,
            metadata=scene.metadata,
        )
    )
    tx_polarization = transmitter_polarizations(scene, device=tx_reference.device)
    rx_polarization = receiver_polarizations(
        scene, device=tx_reference.device, grid=grid
    )
    launch_state = make_launch_state(
        tx_reference, tx_count=len(scene.transmitters), config=config
    )
    endpoint_subpaths = bdpt_endpoint_subpath_state(
        tx_reference,
        tx_power,
        tx_polarization,
        rx_positions,
        rx_polarization,
        launch_state["tx_id"],
        launch_state["light_seed"],
    )
    los_light_state = (
        _reduced_light_endpoint_state(
            tx_reference,
            tx_power,
            tx_polarization,
            rx_positions,
            rx_polarization,
        )
        if "los" in config.components
        else None
    )
    endpoint_connection_samples = None
    endpoint_accumulation = None
    if los_light_state is not None and not scene.structures:
        if config.ad_mode != "none":
            # ADR-022: the endpoint-only (no-structures) LoS fast path threads the
            # frequency and tx_power gradients through the endpoint-connection
            # companion, exactly like _native_los_connection_samples; the
            # accumulate dispatcher below chains the differentiable contribution.
            endpoint_connection_samples = bdpt_endpoint_connection_samples_ad(
                los_light_state,
                endpoint_subpaths["sensor"],
                tx_power,
                frequency=scene.frequency,
                frequency_value=(
                    float(scene.frequency.detach())
                    if isinstance(scene.frequency, torch.Tensor)
                    else float(scene.frequency)
                ),
                samples_per_tx=1,
                max_paths=None,
                mis=config.mis,
                beta=config.power_heuristic_beta,
                strategy_count=1,
            )
            if ledger is not None:
                ledger.add(
                    los_light_state["field_real"],
                    los_light_state["field_imag"],
                    endpoint_subpaths["sensor"]["field_real"],
                    endpoint_subpaths["sensor"]["field_imag"],
                )
        else:
            endpoint_connection_samples = bdpt_endpoint_connection_samples(
                los_light_state,
                endpoint_subpaths["sensor"],
                frequency_hz=float(scene.frequency),
                samples_per_tx=1,
                max_paths=None,
                mis=config.mis,
                beta=config.power_heuristic_beta,
                strategy_count=1,
            )
        endpoint_accumulation = _accumulate_connection_samples(
            config,
            endpoint_connection_samples,
            tx_count=len(scene.transmitters),
            rx_count=int(rx_positions.shape[0]),
            accumulation_strategy=selected_accumulation,
        )
    launch_count = 1

    tx_count = len(scene.transmitters)
    rx_count = int(rx_positions.shape[0])
    endpoint_only = (
        endpoint_accumulation is not None and config.components == frozenset({"los"})
    )
    return _EndpointWorkspace(
        tx_reference=tx_reference,
        tx_power=tx_power,
        rx_positions=rx_positions,
        topology_scene=topology_scene,
        tx_polarization=tx_polarization,
        rx_polarization=rx_polarization,
        launch_state=launch_state,
        endpoint_subpaths=endpoint_subpaths,
        los_light_state=los_light_state,
        endpoint_connection_samples=endpoint_connection_samples,
        endpoint_accumulation=endpoint_accumulation,
        launch_count=launch_count,
        tx_count=tx_count,
        rx_count=rx_count,
        endpoint_only=endpoint_only,
    )
