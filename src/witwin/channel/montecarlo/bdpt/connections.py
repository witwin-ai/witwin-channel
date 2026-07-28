"""Connection-sample builders for the native BDPT solver.

Each builder returns (or streams) connection-sample dictionaries in the
light-major layout consumed by :func:`bdpt_accumulate_connection_samples`.
The enumerated delta reflection/diffraction/coupled builders stay in
``pipeline`` because they depend on the shared ``propagation`` enumerated engine
(ADR-008/ADR-018); this module owns the native LoS, straight transmission and
event-selected shooting connection samplers plus their event-merge helpers.
"""

from __future__ import annotations

from typing import Any

import torch

from witwin.channel.kernels.materials import em_layer_stack_eval
from witwin.channel.montecarlo.bdpt.autograd import (
    bdpt_endpoint_connection_samples_ad,
    bdpt_reflected_light_subpath_state_ad,
    bdpt_transmitted_light_subpath_state_ad,
)
from witwin.channel.kernels.montecarlo import (
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_connection_visibility_inputs,
    bdpt_endpoint_subpath_state,
    bdpt_filter_connection_samples,
    bdpt_reflected_light_subpath_state,
    bdpt_subpath_intersection_inputs,
    bdpt_transmitted_light_subpath_state,
)
from witwin.channel.runtime import _ad_frequency_value
from witwin.channel.kernels.montecarlo import (
    bdpt_reflection_launch_inputs,
    bdpt_sample_directions,
)
from witwin.channel.montecarlo.events.scattering import (
    MASK_SCATTERING,
    local_frames,
    scatter_carried_incident_power,
    scatter_direction_uniforms,
    scattered_subpath_state,
    scattering_nee_connection_samples,
    te_tm_incident_power,
    three_way_rough_probabilities,
    world_to_local,
)
from witwin.channel.montecarlo.events.transmission import (
    event_uniforms,
    layer_csr_view,
    transmission_event_probability,
    unpolarized_power_budgets,
)
from witwin.channel.kernels import geometry as geometry_kernels


_MASK_REFLECTION = 2
_MASK_TRANSMISSION = 8


def _native_los_connection_samples(
    rayd: Any,
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    scene_has_structures: bool,
    frequency_hz: float | torch.Tensor,
    mis: str,
    beta: float,
    strategy_count: int,
    ad: bool = False,
    tx_power: torch.Tensor | None = None,
    frequency_value: float | None = None,
    ledger: object | None = None,
) -> dict[str, torch.Tensor]:
    if ad:
        # ADR-022: the LoS direct connection carries both a frequency gradient
        # (the lambda^2 radiometric factor) and a tx_power gradient (P_src),
        # dispatched natively through the endpoint-connection companion exactly
        # like the mixed-transmission path. The live frequency tensor and the
        # live tx_power leaf feed grad_frequency / grad_tx_power; the host scalar
        # (frequency_value) threads the frozen sampling/pdf path.
        samples = bdpt_endpoint_connection_samples_ad(
            light,
            sensor,
            tx_power,
            frequency=frequency_hz,
            frequency_value=frequency_value,
            samples_per_tx=1,
            max_paths=None,
            mis=mis,
            beta=beta,
            strategy_count=strategy_count,
        )
        if ledger is not None:
            ledger.add(
                light["field_real"],
                light["field_imag"],
                sensor["field_real"],
                sensor["field_imag"],
            )
    else:
        samples = bdpt_endpoint_connection_samples(
            light,
            sensor,
            frequency_hz=frequency_hz,
            samples_per_tx=1,
            max_paths=None,
            mis=mis,
            beta=beta,
            strategy_count=strategy_count,
        )
    if not scene_has_structures:
        return samples
    visibility_inputs = bdpt_endpoint_connection_visibility_inputs(
        light,
        sensor,
        sample_count=int(samples["valid"].shape[0]),
    )
    visible = geometry_kernels.rayd_visibility_forward(
        rayd.require_resource(),
        visibility_inputs["start"],
        visibility_inputs["end"],
        visibility_inputs["active"],
    )[0]
    return bdpt_filter_connection_samples(samples, visible)


def _merge_event_states(
    reflected: dict[str, torch.Tensor],
    transmitted: dict[str, torch.Tensor],
    choose_transmit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Row-wise merge of the two event kernels' outputs.

    Both kernels are evaluated on the full batch and the per-row winner is
    selected, which preserves the tensor layout exactly (equivalent to
    partitioning the hit indices, running each kernel on its partition, and
    scattering back by original index).
    """

    wide = choose_transmit[:, None]
    merged: dict[str, torch.Tensor] = {}
    for key, reflected_value in reflected.items():
        condition = wide if reflected_value.dim() == 2 else choose_transmit
        merged[key] = torch.where(condition, transmitted[key], reflected_value)
    return merged


def _merge_scattered_state(
    merged: dict[str, torch.Tensor],
    scattered: dict[str, torch.Tensor],
    choose_scatter: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Row-wise overlay of the scattered branch onto the reflect/transmit
    merge (same evaluate-everywhere-select-per-row pattern as
    :func:`_merge_event_states`)."""

    wide = choose_scatter[:, None]
    out: dict[str, torch.Tensor] = {}
    for key, merged_value in merged.items():
        condition = wide if merged_value.dim() == 2 else choose_scatter
        out[key] = torch.where(condition, scattered[key], merged_value)
    return out


def _select_surface_events(
    *,
    cos_theta: torch.Tensor,
    material_id: torch.Tensor,
    hit_ok: torch.Tensor,
    material_bundle: dict[str, torch.Tensor],
    layer_csr: dict[str, torch.Tensor],
    runtimes: dict[int, Any],
    frequency_value: float,
    samples: int,
    seed: int,
    tx_index: int,
    bounce: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Three-way (scatter / transmit / reflect) event selection at a surface hit.

    Pure lift of plan section 7.1's frozen event-probability stack: the smooth
    two-way split, the rough three-way budget overlay on rough rows, and the
    single seeded uniform that partitions scatter/transmit/reflect. Returns the
    per-row selection masks plus the probabilities the unbiased weighting reads.
    """

    stack = em_layer_stack_eval(
        cos_theta,
        material_id.clamp_min(0),
        layer_csr["layer_offset"],
        layer_csr["layer_count"],
        layer_csr["layer_thickness_m"],
        layer_csr["layer_eps_r"],
        layer_csr["layer_sigma_e"],
        layer_csr["layer_mu_r"],
        frequency_hz=frequency_value,
    )
    r_eff, t_eff = unpolarized_power_budgets(stack)
    p_transmit = transmission_event_probability(r_eff, t_eff)
    uniforms = event_uniforms(
        int(samples), seed=seed, tx_index=tx_index, depth=bounce, device=device
    )
    if runtimes:
        rough_probs = three_way_rough_probabilities(
            cos_theta,
            material_id,
            material_bundle,
            stack,
            frequency_hz=frequency_value,
        )
        rough = rough_probs["rough"] & hit_ok
        p_scatter = torch.where(
            rough, rough_probs["p_scatter"], torch.zeros_like(p_transmit)
        )
        # Smooth rows keep the exact wave-2 two-way probability;
        # rough rows switch to the three-way budget split.
        p_transmit = torch.where(rough, rough_probs["p_transmit"], p_transmit)
        coherent_amplitude = torch.where(
            rough,
            rough_probs["r_coh_amplitude"],
            torch.ones_like(p_transmit),
        )
    else:
        rough = torch.zeros_like(hit_ok)
        p_scatter = torch.zeros_like(p_transmit)
        coherent_amplitude = torch.ones_like(p_transmit)
    # One uniform partitions the three events: [0, p_s) scatter,
    # [p_s, p_s + p_t) transmit, else reflect. Smooth faces have
    # p_s = 0 exactly, so their transmit test u < p_t is unchanged.
    choose_scatter = hit_ok & (uniforms < p_scatter)
    choose_transmit = (
        hit_ok & ~choose_scatter & (uniforms < p_scatter + p_transmit)
    )
    return {
        "choose_scatter": choose_scatter,
        "choose_transmit": choose_transmit,
        "rough": rough,
        "p_scatter": p_scatter,
        "p_transmit": p_transmit,
        "coherent_amplitude": coherent_amplitude,
    }


def _emit_scatter_nee(
    *,
    rayd: Any,
    sensor: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    hit: dict[str, torch.Tensor],
    merged: dict[str, torch.Tensor],
    choose_scatter: torch.Tensor,
    p_scatter: torch.Tensor,
    material_id: torch.Tensor,
    material_axis_rad: torch.Tensor,
    runtimes: dict[int, Any],
    max_scattering_order: int,
    samples: int,
    seed: int,
    tx_index: int,
    bounce: int,
    device: torch.device,
    scene_diagonal: float,
    frequency_hz: float | torch.Tensor,
    frequency_value: float,
    tx_power: torch.Tensor,
    ad: bool,
    ledger: object | None,
    sample_blocks: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """Scatter branch: local frames, the scattered subpath overlay, and NEE rows.

    Pure lift of the scatter-selected emission block (plan section 7.1, ADR-021
    D4). Appends the NEE connection block (when any) to ``sample_blocks`` and
    returns the (possibly overlaid) merged state, the scattered-valid mask, and
    the count of emitted NEE rows."""

    scattered_valid = torch.zeros_like(choose_scatter)
    nee_rows = 0
    if runtimes and bool(choose_scatter.any()):
        # Local roughness frames from the shading normal flipped
        # toward the incident side (roughness applies to whichever
        # side is illuminated in v1; the store carries front-surface
        # statistics only).
        direction = state["direction"]
        hit_normal = hit["n"]
        normal_flipped = torch.where(
            ((direction * hit_normal).sum(dim=-1) > 0.0)[:, None],
            -hit_normal,
            hit_normal,
        )
        axis_rad = material_axis_rad.index_select(
            0, material_id.clamp_min(0).to(torch.int64)
        )
        frame_t1, frame_t2 = local_frames(normal_flipped, axis_rad)
        wi_world = -direction
        wi_local = world_to_local(wi_world, frame_t1, frame_t2, normal_flipped)
        p_te, p_tm = te_tm_incident_power(
            state["field_real"],
            state["field_imag"],
            direction,
            normal_flipped,
        )
        if int(max_scattering_order) > 1:
            # A subpath that has ALREADY scattered carries no Complex3
            # field (cleared at the previous scatter vertex); its
            # incident power lives in the scalar throughput. Route that
            # unpolarized power into the local TE/TM channels so both
            # the NEE row and the continuation weight see the correct
            # incident power at this vertex. Order 1 never reaches here
            # (no subpath is ever a continued scatter), so the default
            # stays bitwise the field-based decomposition above.
            already_scattered = (
                state["component_mask"] & MASK_SCATTERING
            ) != 0
            carried_te, carried_tm = scatter_carried_incident_power(
                state["throughput_real"], state["throughput_imag"]
            )
            p_te = torch.where(already_scattered, carried_te, p_te)
            p_tm = torch.where(already_scattered, carried_tm, p_tm)
        direction_uniforms = scatter_direction_uniforms(
            int(samples),
            seed=seed,
            tx_index=tx_index,
            depth=bounce,
            device=device,
        )
        scattered = scattered_subpath_state(
            state,
            hit,
            choose_scatter=choose_scatter,
            normal=normal_flipped,
            frame_t1=frame_t1,
            frame_t2=frame_t2,
            wi_local=wi_local,
            p_te=p_te,
            p_tm=p_tm,
            p_scatter=p_scatter,
            material_id=material_id,
            runtimes=runtimes,
            uniforms=direction_uniforms,
            scene_diagonal=scene_diagonal,
            ad=ad,
            ledger=ledger,
        )
        scattered_valid = scattered["valid"]
        merged = _merge_scattered_state(merged, scattered, choose_scatter)
        rows = torch.nonzero(scattered_valid, as_tuple=False).flatten()
        if int(rows.numel()):
            scatter_source_power = state["source_power"].index_select(0, rows)
            if ad:
                # ADR-022 tx_power threading: reattach the live per-tx
                # power's gradient onto the detached native source power
                # for the scatter-selected rows (values bitwise-identical,
                # so the scattering NEE primal is unchanged).
                scatter_tx_id = (
                    state["tx_id"].index_select(0, rows).to(torch.int64)
                )
                live_source_power = tx_power.index_select(0, scatter_tx_id)
                scatter_source_power = scatter_source_power + (
                    live_source_power - live_source_power.detach()
                )
            nee_block = scattering_nee_connection_samples(
                rayd,
                sensor,
                runtimes,
                position=hit["p"].index_select(0, rows),
                normal=normal_flipped.index_select(0, rows),
                frame_t1=frame_t1.index_select(0, rows),
                frame_t2=frame_t2.index_select(0, rows),
                wi_local=wi_local.index_select(0, rows),
                p_te=p_te.index_select(0, rows),
                p_tm=p_tm.index_select(0, rows),
                p_scatter=p_scatter.index_select(0, rows),
                material_id=material_id.index_select(0, rows),
                source_power=scatter_source_power,
                tx_id=state["tx_id"].index_select(0, rows),
                light_depth=scattered["depth"].index_select(0, rows),
                path_length_at_vertex=scattered["path_length"].index_select(
                    0, rows
                ),
                # ADR-015 Part A: hand the live frequency tensor to the
                # radiometric factor under ad; frequency_value stays the
                # host scalar for the sampling/pdf paths.
                frequency_hz=frequency_hz if ad else frequency_value,
                samples=int(samples),
                scene_diagonal=scene_diagonal,
                ad=ad,
                ledger=ledger,
            )
            if nee_block is not None:
                sample_blocks.append(nee_block)
                nee_rows += int(nee_block["valid"].sum())
    return merged, scattered_valid, nee_rows


def _emit_mixed_transmission(
    *,
    rayd: Any,
    sensor: dict[str, torch.Tensor],
    merged: dict[str, torch.Tensor],
    choose_scatter: torch.Tensor,
    emit_mixed_transmission: bool,
    sensor_count: int,
    samples: int,
    tx_power: torch.Tensor,
    frequency_hz: float | torch.Tensor,
    frequency_value: float,
    mis: str,
    beta: float,
    ad: bool,
    ledger: object | None,
    sample_blocks: list[dict[str, torch.Tensor]],
) -> None:
    """Emit the MIXED reflection+transmission endpoint connection (component 5).

    Pure lift of the mixed-transmission emission block (wave 2): connects only the
    reflect-and-transmit subpaths through the native endpoint kernel, filters by
    visibility, and appends the resulting block to ``sample_blocks``."""

    mask = merged["component_mask"]
    mixed = (
        merged["valid"]
        & ~choose_scatter
        & ((mask & _MASK_REFLECTION) != 0)
        & ((mask & _MASK_TRANSMISSION) != 0)
        # Post-scatter subpaths carry no Complex3 field (cleared at
        # the scatter vertex); their |F|^2 = 0 endpoint rows would
        # contribute nothing while contaminating the component-5
        # sample statistics. Their path class (S -> ... -> T) is
        # explicitly not covered in v1 (ADR-021 D4). At order 1 no
        # subpath survives with the scattering bit, so this term is
        # structurally inert for the default.
        & ((mask & MASK_SCATTERING) == 0)
    )
    if emit_mixed_transmission and bool(mixed.any()):
        if ad:
            samples_out = bdpt_endpoint_connection_samples_ad(
                merged,
                sensor,
                tx_power,
                frequency=frequency_hz,
                frequency_value=frequency_value,
                samples_per_tx=int(samples),
                max_paths=None,
                mis=mis,
                beta=beta,
                strategy_count=1,
            )
            if ledger is not None:
                ledger.add(
                    merged["field_real"],
                    merged["field_imag"],
                    sensor["field_real"],
                    sensor["field_imag"],
                )
        else:
            samples_out = bdpt_endpoint_connection_samples(
                merged,
                sensor,
                frequency_hz=frequency_value,
                samples_per_tx=int(samples),
                max_paths=None,
                mis=mis,
                beta=beta,
                strategy_count=1,
            )
        visibility_inputs = bdpt_endpoint_connection_visibility_inputs(
            merged,
            sensor,
            sample_count=int(samples_out["valid"].shape[0]),
        )
        visible = geometry_kernels.rayd_visibility_forward(
            rayd.require_resource(),
            visibility_inputs["start"],
            visibility_inputs["end"],
            visibility_inputs["active"],
        )[0]
        keep = visible & mixed.repeat_interleave(sensor_count)
        sample_blocks.append(bdpt_filter_connection_samples(samples_out, keep))


def _apply_scatter_continuation(
    *,
    merged: dict[str, torch.Tensor],
    scatter_count: torch.Tensor | None,
    choose_scatter: torch.Tensor,
    scattered_valid: torch.Tensor,
    max_scattering_order: int,
) -> torch.Tensor | None:
    """Terminate (order 1) or continue (order > 1) scattered subpaths.

    Pure lift of the continuation/kill logic (ADR-021 D4). Mutates
    ``merged['valid']`` in place and returns the updated scatter-event tally."""

    if scatter_count is None:
        # order 1 (default): scattered subpaths connected above and
        # terminate here; reflection/transmission never follow them.
        merged["valid"] = merged["valid"] & ~choose_scatter
        return None
    # order > 1 (ADR-021 D4): a successfully scattered subpath
    # CONTINUES (its new direction/origin/throughput are already
    # overlaid in ``merged`` by _merge_scattered_state) until it
    # reaches the scatter-event cap. NEE rows were emitted above at
    # this vertex exactly as at order 1. Count only successful
    # scatter events; a subpath that just hit the cap terminates
    # (its NEE is already recorded), and a scatter selection whose
    # direction sample failed is dropped.
    scatter_count = scatter_count + scattered_valid.to(scatter_count.dtype)
    reached_cap = scatter_count >= int(max_scattering_order)
    merged["valid"] = (
        merged["valid"]
        & ~(choose_scatter & ~scattered_valid)
        & ~reached_cap
    )
    return scatter_count


def _transmission_sampled_connection_samples(
    rayd: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    tx_polarization: torch.Tensor,
    rx_positions: torch.Tensor,
    rx_polarization: torch.Tensor,
    sensor: dict[str, torch.Tensor],
    material_bundle: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    samples: int,
    max_depth: int,
    seed: int,
    mis: str,
    beta: float,
    scattering_runtimes: dict[int, Any] | None = None,
    emit_mixed_transmission: bool = True,
    scene_diagonal: float = 0.0,
    max_scattering_order: int = 1,
    ad: bool = False,
    ledger: object | None = None,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    """Shooting-context light subpaths with three-way event selection.

    Implements plan section 7.1. At every surface hit a seeded, reproducible
    uniform selects among the delta specular reflection, the continuous
    Kirchhoff scattering (rough faces only, when ``scattering_runtimes`` is
    provided) and the delta transmission events; the selected branch's field
    is divided by sqrt(p_event) so the power estimator stays unbiased (see
    the inline algebra note).

    Event probabilities: smooth faces keep the wave-2 two-way split
    p_t = T/(R+T) from the native stack budgets BIT-IDENTICALLY (their
    scatter probability is exactly zero, so the same uniform partitions the
    same way); rough faces use the native (R_coh, R_diff, T_bar) budgets
    with the same floor pattern. The rough reflect branch
    additionally multiplies the field by the coherent attenuation C_r so its
    amplitude represents sqrt(R_coh), matching the budget that selected it.

    Contribution routing (never double counts):
    - MIXED reflection+transmission chains connect through the native
      endpoint kernel (component 5), as in wave 2; emitted only when
      ``emit_mixed_transmission`` (the transmission component is requested).
    - Scatter-selected vertices emit torch-side NEE rows (component 6).
      Depth rule (ADR-021 D4, ``max_scattering_order``):
        * order 1 (default, BIT-IDENTICAL): the scattered subpath emits its
          NEE row and TERMINATES; reflection/transmission never follow.
        * order > 1: the scattered subpath CONTINUES in its lobe-sampled
          direction (power divided by ``p_scatter * pdf(wo)`` in
          :func:`scattered_subpath_state`) and may reflect/transmit/scatter
          again, emitting an NEE row at every scatter vertex, until it has
          undergone ``max_scattering_order`` scatter events. A post-scatter
          subpath carries power in the scalar throughput (its Complex3 Jones
          field is cleared at a scatter vertex), so its incident power at a
          further scatter vertex is the unpolarized throughput power.
    - Pure reflection stays with the discrete enumeration and pure
      transmission with the straight endpoint chains.
    """

    device = tx_positions.device
    # ADR-015 Part A: under AD the carrier crosses as a live 0-dim tensor
    # (``frequency_hz``) while the host scalar (``frequency_value``) is read once
    # and threaded to the frozen event-probability stack and every _ad facade.
    # The primal path keeps ``frequency_hz`` a float, so ``frequency_value`` is
    # exactly that float and every call is bitwise the pre-AD behaviour.
    frequency_value = (
        _ad_frequency_value(frequency_hz) if ad else float(frequency_hz)
    )
    layer_csr = layer_csr_view(material_bundle)
    face_material_id = material_bundle["material_id"]
    material_axis_rad = material_bundle["rough_axis_rad"]
    runtimes = scattering_runtimes or {}
    sensor_count = int(sensor["origin"].shape[0])
    sample_blocks: list[dict[str, torch.Tensor]] = []
    transmit_events = 0
    reflect_events = 0
    scatter_events = 0
    nee_rows = 0
    for tx_index in range(int(tx_positions.shape[0])):
        launch_inputs = bdpt_reflection_launch_inputs(
            tx_positions, tx_index=tx_index, sample_count=int(samples)
        )
        ray_d = bdpt_sample_directions(
            int(samples), tx_positions, seed=int(seed) + tx_index * 65537
        )
        state = bdpt_endpoint_subpath_state(
            tx_positions,
            tx_power,
            tx_polarization,
            rx_positions,
            rx_polarization,
            launch_inputs["tx_id"],
            launch_inputs["light_seed"],
        )["light"]
        state["direction"] = ray_d
        ray_inputs = {
            "ray_o": launch_inputs["ray_o"],
            "ray_d": ray_d,
            "ray_tmax": launch_inputs["ray_tmax"],
            "active": launch_inputs["active"],
        }
        # Per-subpath diffuse-scatter event tally for the order cap. Only
        # allocated when multi-order continuation is requested; order 1 keeps
        # the single-bounce terminal rule and never reads it.
        scatter_count = (
            torch.zeros((int(samples),), device=device, dtype=torch.int32)
            if int(max_scattering_order) > 1
            else None
        )
        for bounce in range(max(1, int(max_depth))):
            hit = geometry_kernels.rayd_intersect_forward(
                rayd.require_resource(),
                ray_inputs["ray_o"],
                ray_inputs["ray_d"],
                ray_inputs["ray_tmax"],
                ray_inputs["active"],
            )
            prim = hit["global_prim_id"]
            material_id = face_material_id.index_select(
                0, prim.clamp_min(0).to(torch.int64)
            )
            hit_ok = (
                state["valid"] & (prim >= 0) & (hit["t"] >= 0.0) & (material_id >= 0)
            )
            cos_theta = (
                (state["direction"] * hit["n"]).sum(dim=-1).abs().clamp(1.0e-6, 1.0)
            )
            # The event-probability stack is FROZEN (it drives sampling and MIS,
            # frozen under ADR-022): always the non-AD primal evaluation with the
            # host scalar. Material/layer gradients ride the subpath _ad kernels
            # below, not this selection stack.
            events = _select_surface_events(
                cos_theta=cos_theta,
                material_id=material_id,
                hit_ok=hit_ok,
                material_bundle=material_bundle,
                layer_csr=layer_csr,
                runtimes=runtimes,
                frequency_value=frequency_value,
                samples=samples,
                seed=seed,
                tx_index=tx_index,
                bounce=bounce,
                device=device,
            )
            choose_scatter = events["choose_scatter"]
            choose_transmit = events["choose_transmit"]
            rough = events["rough"]
            p_scatter = events["p_scatter"]
            p_transmit = events["p_transmit"]
            coherent_amplitude = events["coherent_amplitude"]
            if ad:
                if ledger is not None:
                    ledger.add(
                        state["field_real"],
                        state["field_imag"],
                        material_bundle["eps_r"],
                        material_bundle["sigma_e"],
                        material_bundle["thickness"],
                    )
                reflected = bdpt_reflected_light_subpath_state_ad(
                    state,
                    hit,
                    material_gain=material_bundle["gain"],
                    material_valid=material_bundle["valid"],
                    material_eps_r=material_bundle["eps_r"],
                    material_sigma_e=material_bundle["sigma_e"],
                    material_mu_r=material_bundle["mu_r"],
                    material_thickness=material_bundle["thickness"],
                    frequency=frequency_hz,
                    frequency_value=frequency_value,
                )
                if ledger is not None:
                    ledger.add(
                        state["field_real"],
                        state["field_imag"],
                        layer_csr["layer_thickness_m"],
                        layer_csr["layer_eps_r"],
                        layer_csr["layer_sigma_e"],
                    )
                transmitted = bdpt_transmitted_light_subpath_state_ad(
                    state,
                    hit,
                    face_material_id=face_material_id,
                    layer_offset=layer_csr["layer_offset"],
                    layer_count=layer_csr["layer_count"],
                    layer_thickness_m=layer_csr["layer_thickness_m"],
                    layer_eps_r=layer_csr["layer_eps_r"],
                    layer_sigma_e=layer_csr["layer_sigma_e"],
                    layer_mu_r=layer_csr["layer_mu_r"],
                    frequency=frequency_hz,
                    frequency_value=frequency_value,
                )
            else:
                reflected = bdpt_reflected_light_subpath_state(
                    state,
                    hit,
                    material_gain=material_bundle["gain"],
                    material_valid=material_bundle["valid"],
                    material_eps_r=material_bundle["eps_r"],
                    material_sigma_e=material_bundle["sigma_e"],
                    material_mu_r=material_bundle["mu_r"],
                    material_thickness=material_bundle["thickness"],
                    frequency_hz=frequency_value,
                )
                transmitted = bdpt_transmitted_light_subpath_state(
                    state,
                    hit,
                    face_material_id=face_material_id,
                    layer_offset=layer_csr["layer_offset"],
                    layer_count=layer_csr["layer_count"],
                    layer_thickness_m=layer_csr["layer_thickness_m"],
                    layer_eps_r=layer_csr["layer_eps_r"],
                    layer_sigma_e=layer_csr["layer_sigma_e"],
                    layer_mu_r=layer_csr["layer_mu_r"],
                    frequency_hz=frequency_value,
                )
            merged = _merge_event_states(reflected, transmitted, choose_transmit)
            # Unbiased event split: every contribution downstream has the form
            # source_power * |field|^2 * (geometry terms), and branch e was
            # selected with probability p_e, so the POWER must be divided by
            # p_e. Dividing the FIELD (and the real amplitude proxy) by
            # sqrt(p_e) achieves exactly that:
            #   E[|field_e / sqrt(p_e)|^2] = sum_e p_e * |field_e|^2 / p_e
            #                              = sum_e |field_e|^2.
            # source_power is deliberately untouched; scaling it too would
            # double count the correction. The reflect probability is
            # 1 - p_s - p_t (p_s = 0 on smooth faces, reproducing wave 2).
            p_event = torch.where(
                choose_transmit, p_transmit, 1.0 - p_scatter - p_transmit
            )
            inv_amplitude = torch.where(
                merged["valid"],
                torch.rsqrt(p_event.clamp_min(1.0e-4)),
                torch.ones_like(p_event),
            )
            # Rough reflect branch: the native kernel applied the SMOOTH
            # stack Jones (amplitude sqrt(R_bar)); multiplying by C_r turns
            # it into the coherent amplitude sqrt(R_coh) that matches the
            # budget driving its selection probability (contract 6.2).
            reflect_scale = torch.where(
                rough & ~choose_transmit & ~choose_scatter,
                coherent_amplitude,
                torch.ones_like(coherent_amplitude),
            )
            amplitude_scale = inv_amplitude * reflect_scale
            for key in ("throughput_real", "throughput_imag"):
                merged[key] = merged[key] * amplitude_scale
            for key in ("field_real", "field_imag"):
                merged[key] = merged[key] * amplitude_scale[:, None]
            # Reflection/transmission directions are delta events, but the
            # sampled event class still has a discrete probability mass. Keep
            # that mass in the proposal density; canonical enumerated delta
            # paths use a separate unit-mass block.
            for key in ("pdf_forward", "pdf_reverse"):
                merged[key] = torch.where(
                    merged["valid"], merged[key] * p_event, torch.zeros_like(p_event)
                )

            merged, scattered_valid, scatter_nee_rows = _emit_scatter_nee(
                rayd=rayd,
                sensor=sensor,
                state=state,
                hit=hit,
                merged=merged,
                choose_scatter=choose_scatter,
                p_scatter=p_scatter,
                material_id=material_id,
                material_axis_rad=material_axis_rad,
                runtimes=runtimes,
                max_scattering_order=max_scattering_order,
                samples=samples,
                seed=seed,
                tx_index=tx_index,
                bounce=bounce,
                device=device,
                scene_diagonal=scene_diagonal,
                frequency_hz=frequency_hz,
                frequency_value=frequency_value,
                tx_power=tx_power,
                ad=ad,
                ledger=ledger,
                sample_blocks=sample_blocks,
            )
            nee_rows += scatter_nee_rows
            transmit_events += int((choose_transmit & merged["valid"]).sum())
            scatter_events += int(scattered_valid.sum())
            reflect_events += int(
                (~choose_transmit & ~choose_scatter & merged["valid"]).sum()
            )
            _emit_mixed_transmission(
                rayd=rayd,
                sensor=sensor,
                merged=merged,
                choose_scatter=choose_scatter,
                emit_mixed_transmission=emit_mixed_transmission,
                sensor_count=sensor_count,
                samples=samples,
                tx_power=tx_power,
                frequency_hz=frequency_hz,
                frequency_value=frequency_value,
                mis=mis,
                beta=beta,
                ad=ad,
                ledger=ledger,
                sample_blocks=sample_blocks,
            )
            scatter_count = _apply_scatter_continuation(
                merged=merged,
                scatter_count=scatter_count,
                choose_scatter=choose_scatter,
                scattered_valid=scattered_valid,
                max_scattering_order=max_scattering_order,
            )
            if not bool(merged["valid"].any()):
                break
            state = merged
            ray_inputs = bdpt_subpath_intersection_inputs(merged)
    return sample_blocks, {
        "transmit": transmit_events,
        "reflect": reflect_events,
        "scatter": scatter_events,
        "scattering_nee_rows": nee_rows,
    }
