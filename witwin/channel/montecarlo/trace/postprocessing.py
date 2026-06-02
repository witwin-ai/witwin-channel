"""Cell-centered UTD shadow-boundary weights for incoherent Monte Carlo maps."""

from __future__ import annotations

import math

import drjit as dr

from witwin.channel.montecarlo import types as wt
from witwin.channel.core.numerics import arrays
from witwin.channel.core.numerics.constants import (
    DIFFRACTION_MIN_DISTANCE,
    EPS,
    RAY_ORIGIN_BIAS,
)

from witwin.channel.core.geometry.diffraction import wedge_exterior_mask
from witwin.channel.core.kernels.shadow_boundary import ShadowBoundaryKernel
from witwin.channel.core.physics.polarization import normalize_real_with_fallback
from witwin.channel.core.physics.shadow_boundary_policy import ShadowBoundaryBackendPolicy
from witwin.channel.core.physics.wave_math import fresnel_integral
from .diffraction import DiffractionStates
from .diffraction_utd import UTD
from .los import LoS


SHADOW_BOUNDARY_NATIVE_PAIR_THRESHOLD = 1_000_000
_MC_SHADOW_BOUNDARY_POLICY = ShadowBoundaryBackendPolicy(
    small_backend="drjit",
    pair_threshold=SHADOW_BOUNDARY_NATIVE_PAIR_THRESHOLD,
    too_large_message=(
        "shadow_boundary_backend='drjit' is limited to small reference "
        "cases ({n_pairs} edge-cell pairs requested). Use "
        "shadow_boundary_backend='native_candidate' or shadow_boundary_mode='none'."
    ),
    no_native_message=(
        "shadow_boundary_backend='native_candidate' requires the Monte Carlo "
        "native extension with shadow-boundary kernels. Reinstall with "
        "`pip install . --no-deps` or use shadow_boundary_mode='none'."
    ),
    ad_unsupported_message=(
        "shadow_boundary_backend='native_candidate' does not support AD. "
        "Use shadow_boundary_mode='none' for AD runs."
    ),
)
SHADOW_BOUNDARY_ARRAY_KEYS = (
    "incident_shadow_boundary_weight",
    "reflection_shadow_boundary_weight",
    "incident_transition_response_real",
    "incident_transition_response_imag",
    "reflection_transition_response_real",
    "reflection_transition_response_imag",
)
SOURCE_VISIBILITY_SAMPLE_FRACTIONS = (0.02, 0.25, 0.5, 0.75, 0.98)


class ShadowBoundary:
    """Low-overhead geometry weights for power-domain shadow-boundary smoothing."""

    @staticmethod
    def _tile_count(grid, tile_shape: tuple[int, int]) -> int:
        nx, ny = (int(value) for value in grid.grid_shape)
        tile_x, tile_y = (int(value) for value in tile_shape)
        return int(math.ceil(nx / tile_x) * math.ceil(ny / tile_y))

    @staticmethod
    def _metadata(
        *,
        backend: str,
        n_states: int,
        n_cells: int,
        grid,
        config,
        candidate_tiles: int | None = None,
        candidate_pairs: int | None = None,
        candidate_ratio: float | None = None,
        source_total_edges: int | None = None,
        source_visible_edges: int | None = None,
        source_visibility_samples_per_edge: int | None = None,
    ) -> dict[str, object]:
        tile_shape = tuple(int(value) for value in config.shadow_boundary_tile_shape)
        full_pairs = max(1, int(n_states) * int(n_cells))
        if candidate_tiles is None:
            candidate_tiles = (
                int(n_states) * ShadowBoundary._tile_count(grid, tile_shape)
                if backend == "drjit"
                else 0
            )
        if candidate_pairs is None:
            candidate_pairs = int(n_states) * int(n_cells) if backend == "drjit" else 0
        if candidate_ratio is None:
            candidate_ratio = float(candidate_pairs) / float(full_pairs)
        metadata = {
            "backend": str(backend),
            "candidate_tiles": int(candidate_tiles),
            "candidate_pairs": int(candidate_pairs),
            "candidate_ratio": float(candidate_ratio),
            "tile_shape": tile_shape,
            "band_width_wavelengths": float(config.shadow_boundary_band_width_wavelengths),
            "weight_aggregation": "max_weight_weighted_response_average",
            "incident_support": "direct_los_or_matching_first_blocker_surface_group",
        }
        if source_total_edges is not None:
            metadata["source_total_edges"] = int(source_total_edges)
        if source_visible_edges is not None:
            metadata["source_visible_edges"] = int(source_visible_edges)
        if source_visibility_samples_per_edge is not None:
            metadata["source_visibility_samples_per_edge"] = int(
                source_visibility_samples_per_edge
            )
        return metadata

    @staticmethod
    def _zero_result(
        *,
        n_cells: int,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        zero = dr.zeros(wt.Float, n_cells)
        return {
            **{key: zero for key in SHADOW_BOUNDARY_ARRAY_KEYS},
            "_metadata": metadata,
        }

    @staticmethod
    def _resolve_backend(
        *,
        config,
        n_states: int,
        n_cells: int,
        ad_enabled: bool,
    ) -> str:
        return _MC_SHADOW_BOUNDARY_POLICY.resolve(
            requested=str(getattr(config, "shadow_boundary_backend", "drjit")),
            n_pairs=int(n_states) * int(n_cells),
            ad_enabled=bool(ad_enabled),
        )

    @staticmethod
    def _source_visible_edge_mask(
        *,
        states: DiffractionStates,
        scene,
    ):
        n_states = int(dr.width(states.edge_pos.x))
        if n_states <= 0:
            return dr.zeros(wt.Bool, 0)
        edge_hat = normalize_real_with_fallback(
            states.edge_dir,
            wt.Vector3f(0.0, 0.0, 1.0),
        )
        return scene.axial_edge_visible(
            states.source_pos,
            states.edge_pos,
            edge_hat,
            wt.Float(states.edge_line_min),
            wt.Float(states.edge_line_max),
            SOURCE_VISIBILITY_SAMPLE_FRACTIONS,
        )

    @staticmethod
    def _direct_visibility_context(*, scene, tx_pos, grid):
        n_cells = int(grid.n_cells)
        if n_cells <= 0:
            return dr.zeros(wt.Bool, 0), dr.zeros(wt.Int32, 0)
        tri_data = None if scene is None else scene._triangle_runtime()
        if tri_data is None:
            return dr.full(wt.Bool, True, n_cells), dr.full(wt.Int32, -1, n_cells)
        source_pos = arrays.broadcast(tx_pos, n_cells)
        seg_vec = grid.cell_centers - source_pos
        seg_len = dr.norm(seg_vec)
        min_seg_len = wt.Float(2.0 * RAY_ORIGIN_BIAS + EPS)
        active = seg_len > min_seg_len
        seg_dir = seg_vec / (seg_len + EPS)
        ray_origin = source_pos + seg_dir * wt.Float(RAY_ORIGIN_BIAS)
        remaining = dr.maximum(
            seg_len - wt.Float(2.0 * RAY_ORIGIN_BIAS),
            wt.Float(0.0),
        )
        hit, _, prim_idx = scene.intersect_rays_raw_with_prim(
            ray_origin,
            seg_dir,
            active,
            tmax=remaining,
        )
        blocker_group = scene.triangle_group_id(wt.Int32(prim_idx))
        visible = active & ~hit
        blocker_group = dr.select(hit, blocker_group, wt.Int32(-1))
        return visible, blocker_group

    @staticmethod
    def _edge_adjacent_groups(*, states: DiffractionStates, scene):
        n_states = int(dr.width(states.edge_pos.x))
        tri_data = None if scene is None else scene._triangle_runtime()
        if n_states <= 0:
            return dr.zeros(wt.Int32, 0), dr.zeros(wt.Int32, 0)
        if tri_data is None:
            return dr.full(wt.Int32, -1, n_states), dr.full(wt.Int32, -1, n_states)
        return (
            scene.triangle_group_id(wt.Int32(states.adjacent_face0)),
            scene.triangle_group_id(wt.Int32(states.adjacent_face1)),
        )

    @staticmethod
    def _finite_edge_factor(
        *,
        source_pos,
        target_pos,
        edge_pos,
        edge_dir,
        edge_line_min,
        edge_line_max,
        k: float,
    ):
        width = int(dr.width(target_pos.x))
        edge_pos_b = arrays.broadcast(edge_pos, width)
        edge_hat = normalize_real_with_fallback(
            arrays.broadcast(edge_dir, width),
            wt.Vector3f(0.0, 0.0, 1.0),
        )
        source_pos_b = arrays.broadcast(source_pos, width)
        line_min = arrays.broadcast(edge_line_min, width)
        line_max = arrays.broadcast(edge_line_max, width)

        source_axial = dr.dot(source_pos_b - edge_pos_b, edge_hat)
        target_axial = dr.dot(target_pos - edge_pos_b, edge_hat)
        source_to_edge = edge_pos_b - source_pos_b
        edge_to_target = target_pos - edge_pos_b
        source_proj = source_to_edge - dr.dot(source_to_edge, edge_hat) * edge_hat
        target_proj = edge_to_target - dr.dot(edge_to_target, edge_hat) * edge_hat
        s_prime_proj = dr.norm(source_proj) + wt.Float(EPS)
        s_proj = dr.norm(target_proj) + wt.Float(EPS)
        stationary_u = (
            s_prime_proj * target_axial + s_proj * source_axial
        ) / (s_proj + s_prime_proj + wt.Float(EPS))
        source_offset = stationary_u - source_axial
        target_offset = target_axial - stationary_u
        source_range = dr.sqrt(
            s_prime_proj * s_prime_proj + source_offset * source_offset + wt.Float(EPS)
        )
        target_range = dr.sqrt(
            s_proj * s_proj + target_offset * target_offset + wt.Float(EPS)
        )
        curvature = (
            s_prime_proj * s_prime_proj
            / (source_range * source_range * source_range + wt.Float(EPS))
            + s_proj * s_proj
            / (target_range * target_range * target_range + wt.Float(EPS))
        )
        scale = dr.sqrt(dr.maximum(wt.Float(k) * curvature, wt.Float(EPS)) / dr.pi)
        delta_f = (
            fresnel_integral(scale * (line_max - stationary_u))
            - fresnel_integral(scale * (line_min - stationary_u))
        )
        finite_factor = wt.Complex2f(0.5, 0.5) * dr.conj(delta_f)
        finite_scale = dr.sqrt(
            dr.maximum(arrays.complex_abs_sqr(finite_factor), wt.Float(0.0))
        )
        finite_scale = dr.minimum(finite_scale, wt.Float(1.0))
        finite_scale = dr.select(dr.isfinite(finite_scale), finite_scale, wt.Float(0.0))
        line_length_valid = (line_max - line_min) > wt.Float(EPS)
        finite_factor = dr.select(
            line_length_valid,
            finite_factor,
            arrays.complex_zero(width),
        )
        finite_scale = dr.select(line_length_valid, finite_scale, wt.Float(0.0))
        line_center = wt.Float(0.5) * (line_min + line_max)
        line_half_length = wt.Float(0.5) * (line_max - line_min)
        stationary_on_segment = (
            line_length_valid
            & (stationary_u >= line_min - wt.Float(EPS))
            & (stationary_u <= line_max + wt.Float(EPS))
        )
        interior_weight = (
            line_half_length - dr.abs(stationary_u - line_center)
        ) / dr.maximum(line_half_length, wt.Float(EPS))
        interior_weight = dr.clip(interior_weight, wt.Float(0.0), wt.Float(1.0))
        endpoint_floor = wt.Float(0.25)
        support_weight = endpoint_floor + (
            wt.Float(1.0) - endpoint_floor
        ) * interior_weight
        support_weight = dr.select(
            stationary_on_segment,
            support_weight,
            wt.Float(0.0),
        )
        return finite_factor, finite_scale, support_weight

    @staticmethod
    def _accumulate_state_weights(
        *,
        states: DiffractionStates,
        grid,
        k: float,
        direct_los_visible=None,
        direct_blocker_group=None,
        edge_adjacent_group0=None,
        edge_adjacent_group1=None,
    ) -> dict[str, object]:
        n_cells = int(grid.n_cells)
        n_states = int(dr.width(states.edge_pos.x))
        if direct_los_visible is None:
            direct_los_visible = dr.full(wt.Bool, True, n_cells)
        if direct_blocker_group is None:
            direct_blocker_group = dr.full(wt.Int32, -1, n_cells)
        if edge_adjacent_group0 is None:
            edge_adjacent_group0 = states.adjacent_face0
        if edge_adjacent_group1 is None:
            edge_adjacent_group1 = states.adjacent_face1
        incident_weight = dr.zeros(wt.Float, n_cells)
        reflection_weight = dr.zeros(wt.Float, n_cells)
        incident_weight_sum = dr.zeros(wt.Float, n_cells)
        reflection_weight_sum = dr.zeros(wt.Float, n_cells)
        incident_response_real = dr.zeros(wt.Float, n_cells)
        incident_response_imag = dr.zeros(wt.Float, n_cells)
        reflection_response_real = dr.zeros(wt.Float, n_cells)
        reflection_response_imag = dr.zeros(wt.Float, n_cells)
        if n_cells <= 0 or n_states <= 0:
            return {
                "incident_shadow_boundary_weight": incident_weight,
                "reflection_shadow_boundary_weight": reflection_weight,
                "incident_transition_response_real": incident_response_real,
                "incident_transition_response_imag": incident_response_imag,
                "reflection_transition_response_real": reflection_response_real,
                "reflection_transition_response_imag": reflection_response_imag,
            }

        for state_index in range(n_states):
            state = states.gather(wt.UInt32(state_index))
            edge_pos = arrays.broadcast(state.edge_pos, n_cells)
            edge_dir = arrays.broadcast(state.edge_dir, n_cells)
            n0 = arrays.broadcast(state.n0, n_cells)
            nn = arrays.broadcast(state.n_face_n, n_cells)
            source_pos = arrays.broadcast(state.source_pos, n_cells)
            incident_dir = edge_pos - source_pos
            flip = dr.dot(incident_dir, n0) > wt.Float(0.0)
            oriented_edge_dir = dr.select(flip, -edge_dir, edge_dir)
            oriented_n0 = dr.select(flip, nn, n0)
            oriented_nn = dr.select(flip, n0, nn)

            geo = UTD.setup_edge_geometry(
                source_pos,
                edge_pos,
                oriented_edge_dir,
                oriented_n0,
                oriented_nn,
                grid.cell_centers,
                state.wedge_n,
            )
            coeffs = UTD.diffraction_coefficients(
                geo.phi,
                geo.phi_prime,
                geo.wedge_n_b,
                geo.s,
                geo.s_prime,
                geo.sin_beta0,
                k,
                geo.field_valid,
                None,
                geo.width,
            )
            target_exterior = wedge_exterior_mask(
                grid.cell_centers - edge_pos,
                oriented_edge_dir,
                oriented_n0,
                oriented_nn,
            )
            finite_factor, finite_scale, finite_segment_support_weight = (
                ShadowBoundary._finite_edge_factor(
                    source_pos=state.source_pos,
                    target_pos=grid.cell_centers,
                    edge_pos=state.edge_pos,
                    edge_dir=state.edge_dir,
                    edge_line_min=state.edge_line_min,
                    edge_line_max=state.edge_line_max,
                    k=k,
                )
            )
            support = (
                target_exterior
                & coeffs.field_valid
                & coeffs.pole_safe
                & (geo.s > wt.Float(DIFFRACTION_MIN_DISTANCE))
                & (geo.s_prime > wt.Float(DIFFRACTION_MIN_DISTANCE))
                & (geo.wedge_n_b > wt.Float(1.01))
            )
            finite_amplitude = dr.sqrt(dr.maximum(finite_scale, wt.Float(0.0)))
            finite_weight = finite_amplitude * finite_segment_support_weight
            response_weight = dr.select(
                finite_segment_support_weight > wt.Float(0.0),
                finite_amplitude,
                wt.Float(0.0),
            )
            adjacent_group0 = arrays.broadcast(
                dr.gather(wt.Int32, edge_adjacent_group0, wt.UInt32(state_index)),
                n_cells,
            )
            adjacent_group1 = arrays.broadcast(
                dr.gather(wt.Int32, edge_adjacent_group1, wt.UInt32(state_index)),
                n_cells,
            )
            blocker_matches_edge = (
                ((direct_blocker_group == adjacent_group0) & (adjacent_group0 >= wt.Int32(0)))
                | ((direct_blocker_group == adjacent_group1) & (adjacent_group1 >= wt.Int32(0)))
            )
            incident_support = support & (direct_los_visible | blocker_matches_edge)
            edge_incident = dr.select(
                incident_support,
                coeffs.incident_transition_weight * finite_weight,
                wt.Float(0.0),
            )
            edge_incident_response = dr.select(
                incident_support,
                coeffs.incident_transition_weight * response_weight,
                wt.Float(0.0),
            )
            edge_reflection = dr.select(
                support,
                coeffs.reflection_transition_weight * finite_weight,
                wt.Float(0.0),
            )
            edge_reflection_response = dr.select(
                support,
                coeffs.reflection_transition_weight * response_weight,
                wt.Float(0.0),
            )
            incident_response = finite_factor * coeffs.incident_transition_response
            reflection_response = finite_factor * coeffs.reflection_transition_response
            incident_response_real = (
                incident_response_real + edge_incident_response * incident_response.real
            )
            incident_response_imag = (
                incident_response_imag + edge_incident_response * incident_response.imag
            )
            reflection_response_real = (
                reflection_response_real + edge_reflection_response * reflection_response.real
            )
            reflection_response_imag = (
                reflection_response_imag + edge_reflection_response * reflection_response.imag
            )
            incident_weight_sum = incident_weight_sum + edge_incident_response
            reflection_weight_sum = reflection_weight_sum + edge_reflection_response
            incident_weight = dr.maximum(incident_weight, edge_incident)
            reflection_weight = dr.maximum(reflection_weight, edge_reflection)

        safe_incident_weight = dr.maximum(incident_weight_sum, wt.Float(1.0e-6))
        safe_reflection_weight = dr.maximum(reflection_weight_sum, wt.Float(1.0e-6))
        incident_response_real = dr.select(
            incident_weight_sum > wt.Float(1.0e-6),
            incident_response_real / safe_incident_weight,
            wt.Float(0.0),
        )
        incident_response_imag = dr.select(
            incident_weight_sum > wt.Float(1.0e-6),
            incident_response_imag / safe_incident_weight,
            wt.Float(0.0),
        )
        reflection_response_real = dr.select(
            reflection_weight_sum > wt.Float(1.0e-6),
            reflection_response_real / safe_reflection_weight,
            wt.Float(0.0),
        )
        reflection_response_imag = dr.select(
            reflection_weight_sum > wt.Float(1.0e-6),
            reflection_response_imag / safe_reflection_weight,
            wt.Float(0.0),
        )
        incident_weight = dr.clip(incident_weight, wt.Float(0.0), wt.Float(1.0))
        reflection_weight = dr.clip(reflection_weight, wt.Float(0.0), wt.Float(1.0))
        return {
            "incident_shadow_boundary_weight": incident_weight,
            "reflection_shadow_boundary_weight": reflection_weight,
            "incident_transition_response_real": incident_response_real,
            "incident_transition_response_imag": incident_response_imag,
            "reflection_transition_response_real": reflection_response_real,
            "reflection_transition_response_imag": reflection_response_imag,
        }

    @staticmethod
    def transition_weights_for_grid(
        *,
        scene,
        tx_pos,
        grid,
        config,
        edge_indices=None,
        ad_enabled: bool = False,
    ) -> dict[str, object]:
        n_cells = int(grid.n_cells)
        edge_runtime = scene._selected_edge_runtime()
        n_edges = 0 if edge_runtime is None else int(edge_runtime.get("n_edges", 0))
        if n_cells <= 0 or n_edges <= 0:
            return ShadowBoundary._zero_result(
                n_cells=n_cells,
                metadata=ShadowBoundary._metadata(
                    backend="none",
                    n_states=0,
                    n_cells=n_cells,
                    grid=grid,
                    config=config,
                    source_total_edges=n_edges,
                    source_visible_edges=0,
                    source_visibility_samples_per_edge=len(
                        SOURCE_VISIBILITY_SAMPLE_FRACTIONS
                    ),
                ),
            )
        if edge_indices is None:
            edge_indices = dr.arange(wt.UInt32, n_edges)
        elif int(dr.width(edge_indices)) <= 0:
            return ShadowBoundary._zero_result(
                n_cells=n_cells,
                metadata=ShadowBoundary._metadata(
                    backend="none",
                    n_states=0,
                    n_cells=n_cells,
                    grid=grid,
                    config=config,
                    source_total_edges=0,
                    source_visible_edges=0,
                    source_visibility_samples_per_edge=len(
                        SOURCE_VISIBILITY_SAMPLE_FRACTIONS
                    ),
                ),
            )
        source_candidate_edges = int(dr.width(edge_indices))
        states = DiffractionStates.from_edge_indices(
            tx_pos=tx_pos,
            edge_idx=wt.UInt32(edge_indices),
            scene=scene,
            config=config,
        )
        if states is None or int(dr.width(states.edge_pos.x)) <= 0:
            return ShadowBoundary._zero_result(
                n_cells=n_cells,
                metadata=ShadowBoundary._metadata(
                    backend="none",
                    n_states=0,
                    n_cells=n_cells,
                    grid=grid,
                    config=config,
                    source_total_edges=source_candidate_edges,
                    source_visible_edges=0,
                    source_visibility_samples_per_edge=len(
                        SOURCE_VISIBILITY_SAMPLE_FRACTIONS
                    ),
                ),
            )
        source_total_edges = int(dr.width(states.edge_pos.x))
        source_visible_mask = ShadowBoundary._source_visible_edge_mask(
            states=states,
            scene=scene,
        )
        dr.eval(source_visible_mask)
        source_visible_slots = dr.compress(source_visible_mask)
        source_visible_edges = int(dr.width(source_visible_slots))
        if source_visible_edges <= 0:
            return ShadowBoundary._zero_result(
                n_cells=n_cells,
                metadata=ShadowBoundary._metadata(
                    backend="none",
                    n_states=0,
                    n_cells=n_cells,
                    grid=grid,
                    config=config,
                    source_total_edges=source_total_edges,
                    source_visible_edges=0,
                    source_visibility_samples_per_edge=len(
                        SOURCE_VISIBILITY_SAMPLE_FRACTIONS
                    ),
                ),
            )
        states = states.gather(wt.UInt32(source_visible_slots))
        n_states = int(dr.width(states.edge_pos.x))
        backend = ShadowBoundary._resolve_backend(
            config=config,
            n_states=n_states,
            n_cells=n_cells,
            ad_enabled=bool(ad_enabled),
        )
        if backend == "native_candidate":
            direct_los_visible, direct_blocker_group = (
                ShadowBoundary._direct_visibility_context(
                    scene=scene,
                    tx_pos=tx_pos,
                    grid=grid,
                )
            )
            edge_adjacent_group0, edge_adjacent_group1 = ShadowBoundary._edge_adjacent_groups(
                states=states,
                scene=scene,
            )
            result = ShadowBoundaryKernel.accumulate(
                states=states,
                grid=grid,
                k=float(config.k),
                wavelength=float(config.wavelength),
                tile_shape=config.shadow_boundary_tile_shape,
                band_width_wavelengths=float(config.shadow_boundary_band_width_wavelengths),
                max_candidate_factor=float(config.shadow_boundary_max_candidate_factor),
                direct_los_visible=direct_los_visible,
                direct_blocker_group=direct_blocker_group,
                edge_adjacent_group0=edge_adjacent_group0,
                edge_adjacent_group1=edge_adjacent_group1,
            )
            result["_metadata"].update({
                "source_total_edges": source_total_edges,
                "source_visible_edges": source_visible_edges,
                "source_visibility_samples_per_edge": len(
                    SOURCE_VISIBILITY_SAMPLE_FRACTIONS
                ),
            })
            return result

        direct_los_visible, direct_blocker_group = ShadowBoundary._direct_visibility_context(
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
        )
        edge_adjacent_group0, edge_adjacent_group1 = ShadowBoundary._edge_adjacent_groups(
            states=states,
            scene=scene,
        )
        result = ShadowBoundary._accumulate_state_weights(
            states=states,
            grid=grid,
            k=float(config.k),
            direct_los_visible=direct_los_visible,
            direct_blocker_group=direct_blocker_group,
            edge_adjacent_group0=edge_adjacent_group0,
            edge_adjacent_group1=edge_adjacent_group1,
        )
        result["_metadata"] = ShadowBoundary._metadata(
            backend="drjit",
            n_states=n_states,
            n_cells=n_cells,
            grid=grid,
            config=config,
            source_total_edges=source_total_edges,
            source_visible_edges=source_visible_edges,
            source_visibility_samples_per_edge=len(
                SOURCE_VISIBILITY_SAMPLE_FRACTIONS
            ),
        )
        return result

    @staticmethod
    def accumulate_into_diagnostics(
        *,
        weighted_diagnostics: dict,
        scene,
        tx_pos,
        grid,
        config,
        edge_indices=None,
        ad_enabled: bool = False,
    ) -> None:
        weights = ShadowBoundary.transition_weights_for_grid(
            scene=scene,
            tx_pos=tx_pos,
            grid=grid,
            config=config,
            edge_indices=edge_indices,
            ad_enabled=ad_enabled,
        )
        weighted_diagnostics["shadow_boundary_runtime"] = dict(weights.get("_metadata", {}))
        inc = weighted_diagnostics["incoherent"]
        inc["incident_shadow_boundary_weight"] = dr.maximum(
            inc["incident_shadow_boundary_weight"],
            weights["incident_shadow_boundary_weight"],
        )
        inc["reflection_shadow_boundary_weight"] = dr.maximum(
            inc["reflection_shadow_boundary_weight"],
            weights["reflection_shadow_boundary_weight"],
        )
        for key in (
            "incident_transition_response_real",
            "incident_transition_response_imag",
            "reflection_transition_response_real",
            "reflection_transition_response_imag",
        ):
            inc[key] = weights[key]
        continued_power, _ = LoS.power_to_targets(
            tx_pos=tx_pos,
            target_pos=grid.cell_centers,
            config=config,
        )
        inc["continued_incident_power"] = continued_power


__all__ = ["ShadowBoundary"]
