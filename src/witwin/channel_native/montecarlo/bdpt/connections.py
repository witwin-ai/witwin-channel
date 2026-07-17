"""Connection-sample builders for the native BDPT solver.

Each builder returns (or streams) connection-sample dictionaries in the
light-major layout consumed by :func:`bdpt_accumulate_connection_samples`.
The enumerated delta reflection/coupled builders stay in ``pipeline`` because
they depend on the shared ``propagation`` enumerated engine (ADR-008); this
module owns the native LoS, diffraction, straight transmission and
event-selected shooting connection samplers plus their event-merge helpers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from witwin.channel_native.core.diffraction_geometry import (
    cached_diffraction_edge_geometry as _cached_diffraction_edge_geometry,
)
from witwin.channel_native.materials.kernels.functional import em_layer_stack_eval
from witwin.channel_native.montecarlo.bdpt.kernels.paths import (
    bdpt_diffraction_point_connection_samples,
    bdpt_endpoint_connection_samples,
    bdpt_endpoint_connection_visibility_inputs,
    bdpt_endpoint_subpath_state,
    bdpt_filter_connection_samples,
    bdpt_reflected_light_subpath_state,
    bdpt_subpath_intersection_inputs,
    bdpt_transmitted_light_subpath_state,
)
from witwin.channel_native.montecarlo.bdpt.kernels.sampling import (
    bdpt_diffraction_state_pack,
    bdpt_reflection_launch_inputs,
    bdpt_sample_directions,
    bdpt_selected_edge_indices,
)
from witwin.channel_native.montecarlo.events.scattering import (
    local_frames,
    scatter_direction_uniforms,
    scattered_subpath_state,
    scattering_nee_connection_samples,
    te_tm_incident_power,
    three_way_rough_probabilities,
    world_to_local,
)
from witwin.channel_native.montecarlo.events.transmission import (
    event_uniforms,
    layer_csr_view,
    straight_transmission_chains,
    transmission_event_probability,
    unpolarized_power_budgets,
)
from witwin.channel_native.propagation.geometry.kernels import bridge as geometry_bridge
from witwin.channel_native.scene.models import Scene


_LIGHT_SPEED_M_PER_S = 299_792_458.0
_TRANSMISSION_COMPONENT_ID = 5
_MASK_REFLECTION = 2
_MASK_TRANSMISSION = 8


def _native_los_connection_samples(
    raydn: Any,
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    *,
    scene_has_structures: bool,
    frequency_hz: float,
    mis: str,
    beta: float,
    strategy_count: int,
) -> dict[str, torch.Tensor]:
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
    visible = geometry_bridge.raydn_visibility_forward(
        raydn.require_handle(),
        visibility_inputs["start"],
        visibility_inputs["end"],
        visibility_inputs["active"],
    )[0]
    return bdpt_filter_connection_samples(samples, visible)


def _diffraction_sample_split(sample_count: int, *, mis: str) -> tuple[int, int, int]:
    if mis == "none":
        return int(sample_count), 0, 0
    direct = (int(sample_count) + 2) // 3
    keller = (int(sample_count) + 1) // 3
    return direct, keller, 0


def _diffraction_strategy_count(direct_samples: int, keller_samples: int) -> int:
    return (1 if direct_samples > 0 else 0) + (1 if keller_samples > 0 else 0)


def _native_diffraction_point_connection_samples(
    scene: Scene,
    raydn: Any,
    tx_positions: torch.Tensor,
    tx_power: torch.Tensor,
    rx_positions: torch.Tensor,
    material_tensors: tuple[torch.Tensor, ...],
    *,
    samples: int,
    seed: int,
    mis: str,
    beta: float,
) -> Iterator[dict[str, torch.Tensor]]:
    _eps_r, _sigma_e, _mu_r, material_gain, material_valid, _thickness = (
        material_tensors
    )
    edge_geometry = _cached_diffraction_edge_geometry(raydn)
    (
        selected,
        edge_pos,
        edge_dir,
        _lengths,
        line_min,
        line_max,
        n0,
        n1,
        face0,
        face1,
        exterior_angle,
    ) = edge_geometry
    edge_indices = bdpt_selected_edge_indices(selected)
    wavelength = _LIGHT_SPEED_M_PER_S / float(scene.frequency)
    direct_samples, keller_samples, _suffix_samples = _diffraction_sample_split(
        int(samples), mis=mis
    )
    for tx_index in range(int(tx_positions.shape[0])):
        states = bdpt_diffraction_state_pack(
            edge_indices,
            edge_pos,
            edge_dir,
            line_min,
            line_max,
            n0,
            n1,
            face0,
            face1,
            exterior_angle,
            tx_positions[tx_index],
            tx_power[tx_index],
        )
        state_count = int(states[0].shape[0])
        if state_count <= 0:
            continue
        for rx_start in range(0, int(rx_positions.shape[0]), 64):
            rx_end = min(rx_start + 64, int(rx_positions.shape[0]))
            exported = bdpt_diffraction_point_connection_samples(
                rx_positions[rx_start:rx_end],
                states,
                material_gain,
                material_valid,
                tx_index=tx_index,
                state_count=state_count,
                direct_samples=int(direct_samples),
                keller_samples=int(keller_samples),
                seed=int(seed) + tx_index * 104729,
                wavelength=float(wavelength),
                mis=mis,
                beta=beta,
                strategy_count=_diffraction_strategy_count(
                    direct_samples, keller_samples
                ),
            )
            samples_out = exported["samples"]
            if not isinstance(samples_out, dict):
                raise RuntimeError(
                    "native BDPT diffraction point sampler returned invalid samples"
                )
            if rx_start:
                samples_out["rx_id"].add_(rx_start)
                samples_out["grid_linear_id"].add_(rx_start)
                samples_out["topology"][:, 1].add_(rx_start)
            visible_source = geometry_bridge.raydn_visibility_forward(
                raydn.require_handle(),
                exported["source_start"],
                exported["source_end"],
                exported["visibility_active"],
            )[0]
            filtered = bdpt_filter_connection_samples(samples_out, visible_source)
            visible_target = geometry_bridge.raydn_visibility_forward(
                raydn.require_handle(),
                exported["target_start"],
                exported["target_end"],
                exported["visibility_active"],
            )[0]
            yield bdpt_filter_connection_samples(filtered, visible_target)


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


def _transmission_straight_connection_samples(
    raydn: Any,
    light: dict[str, torch.Tensor],
    sensor: dict[str, torch.Tensor],
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    material_bundle: dict[str, torch.Tensor],
    *,
    frequency_hz: float,
    max_depth: int,
    scene_diagonal: float,
    mis: str,
    beta: float,
) -> tuple[dict[str, torch.Tensor] | None, int]:
    """Exact pure-transmission Tx->Rx chains (endpoint-connection context).

    Specular thin_sheet transmission never bends the ray (parallel-plate
    exit), so every pure-transmission path topology IS the straight Tx->Rx
    segment. Marching that segment yields the per-pair power transmittance
    product, which scales the analytic endpoint-connection LoS contribution
    and is reclassified as the exclusive transmission path class (component
    id 5). Like the discrete reflection enumeration, these chains carry unit
    bidirectional mass (mis weight 1). Pairs whose segment crosses no wall
    belong to the los class and are filtered out here; a vacuum wall has unit
    power transmittance, so the transmission component reproduces the
    unobstructed LoS value exactly.
    """

    rx_count = int(rx_positions.shape[0])
    if int(tx_positions.shape[0]) == 0 or rx_count == 0:
        return None, 0
    samples = bdpt_endpoint_connection_samples(
        light,
        sensor,
        frequency_hz=frequency_hz,
        samples_per_tx=1,
        max_paths=None,
        mis=mis,
        beta=beta,
        strategy_count=1,
    )
    layer_csr = layer_csr_view(material_bundle)
    transmittance_rows = []
    penetrated_rows = []
    wall_rows = []
    for tx_index in range(int(tx_positions.shape[0])):
        origins = tx_positions[tx_index].unsqueeze(0).repeat(rx_count, 1)
        chain = straight_transmission_chains(
            raydn,
            origins,
            rx_positions,
            face_material_id=material_bundle["material_id"],
            layer_csr=layer_csr,
            frequency_hz=frequency_hz,
            max_depth=max_depth,
            scene_diagonal=scene_diagonal,
        )
        transmittance_rows.append(chain["transmittance"])
        penetrated_rows.append(chain["penetrated"])
        wall_rows.append(chain["wall_count"])
    # Connection rows are light-major (light_index * sensor_count + sensor)
    # and the reduced light state carries exactly one row per transmitter, so
    # the concatenated per-tx march aligns 1:1 with the connection table.
    transmittance = torch.cat(transmittance_rows, dim=0)
    penetrated = torch.cat(penetrated_rows, dim=0)
    wall_count = torch.cat(wall_rows, dim=0)
    samples["contribution"] = samples["contribution"] * transmittance
    samples = bdpt_filter_connection_samples(samples, penetrated)
    chain_count = int(samples["valid"].sum())
    if chain_count == 0:
        return None, 0
    component_id = torch.where(
        samples["valid"],
        torch.full_like(samples["component_id"], _TRANSMISSION_COMPONENT_ID),
        samples["component_id"],
    )
    light_depth = torch.where(samples["valid"], wall_count, samples["light_depth"])
    topology = samples["topology"].clone()
    topology[:, 2] = component_id
    topology[:, 3] = light_depth
    samples["component_id"] = component_id
    samples["light_depth"] = light_depth
    samples["topology"] = topology
    return samples, chain_count


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


def _transmission_sampled_connection_samples(
    raydn: Any,
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
    same way); rough faces use the (R_coh, R_diff, T_bar) budgets from
    scattering.energy with the same floor pattern. The rough reflect branch
    additionally multiplies the field by the coherent attenuation C_r so its
    amplitude represents sqrt(R_coh), matching the budget that selected it.

    Contribution routing (never double counts):
    - MIXED reflection+transmission chains connect through the native
      endpoint kernel (component 5), as in wave 2; emitted only when
      ``emit_mixed_transmission`` (the transmission component is requested).
    - Scatter-selected vertices emit torch-side NEE rows (component 6) and
      then TERMINATE (v1 single-bounce rule; reflection/transmission never
      follow a scattering event).
    - Pure reflection stays with the discrete enumeration and pure
      transmission with the straight endpoint chains.
    """

    device = tx_positions.device
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
        for bounce in range(max(1, int(max_depth))):
            hit = geometry_bridge.bdpt_intersect_forward(
                raydn.require_handle(),
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
            stack = em_layer_stack_eval(
                cos_theta,
                material_id.clamp_min(0),
                layer_csr["layer_offset"],
                layer_csr["layer_count"],
                layer_csr["layer_thickness_m"],
                layer_csr["layer_eps_r"],
                layer_csr["layer_sigma_e"],
                layer_csr["layer_mu_r"],
                frequency_hz=frequency_hz,
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
                    frequency_hz=float(frequency_hz),
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
            reflected = bdpt_reflected_light_subpath_state(
                state,
                hit,
                material_gain=material_bundle["gain"],
                material_valid=material_bundle["valid"],
                material_eps_r=material_bundle["eps_r"],
                material_sigma_e=material_bundle["sigma_e"],
                material_mu_r=material_bundle["mu_r"],
                material_thickness=material_bundle["thickness"],
                frequency_hz=frequency_hz,
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
                frequency_hz=frequency_hz,
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

            scattered_valid = torch.zeros_like(choose_scatter)
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
                )
                scattered_valid = scattered["valid"]
                merged = _merge_scattered_state(merged, scattered, choose_scatter)
                rows = torch.nonzero(scattered_valid, as_tuple=False).flatten()
                if int(rows.numel()):
                    nee_block = scattering_nee_connection_samples(
                        raydn,
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
                        source_power=state["source_power"].index_select(0, rows),
                        tx_id=state["tx_id"].index_select(0, rows),
                        light_depth=scattered["depth"].index_select(0, rows),
                        path_length_at_vertex=scattered["path_length"].index_select(
                            0, rows
                        ),
                        frequency_hz=float(frequency_hz),
                        samples=int(samples),
                        scene_diagonal=scene_diagonal,
                    )
                    if nee_block is not None:
                        sample_blocks.append(nee_block)
                        nee_rows += int(nee_block["valid"].sum())
            transmit_events += int((choose_transmit & merged["valid"]).sum())
            scatter_events += int(scattered_valid.sum())
            reflect_events += int(
                (~choose_transmit & ~choose_scatter & merged["valid"]).sum()
            )
            mask = merged["component_mask"]
            mixed = (
                merged["valid"]
                & ~choose_scatter
                & ((mask & _MASK_REFLECTION) != 0)
                & ((mask & _MASK_TRANSMISSION) != 0)
            )
            if emit_mixed_transmission and bool(mixed.any()):
                samples_out = bdpt_endpoint_connection_samples(
                    merged,
                    sensor,
                    frequency_hz=frequency_hz,
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
                visible = geometry_bridge.raydn_visibility_forward(
                    raydn.require_handle(),
                    visibility_inputs["start"],
                    visibility_inputs["end"],
                    visibility_inputs["active"],
                )[0]
                keep = visible & mixed.repeat_interleave(sensor_count)
                sample_blocks.append(bdpt_filter_connection_samples(samples_out, keep))
            # v1 single-bounce rule: scattered subpaths connected above and
            # terminate here; reflection/transmission never follow them.
            merged["valid"] = merged["valid"] & ~choose_scatter
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
