from __future__ import annotations

import drjit as dr
from witwin.channel import types as wt

from witwin.channel._native.channel_utils import NativeExtension


class ShadowBoundaryKernel:
    """Static namespace for native shadow-boundary candidate smoothing."""

    FUNCTION_NAME = "shadow_boundary_candidate_accumulate"
    _REQUIRED = (FUNCTION_NAME,)

    @staticmethod
    def require():
        return NativeExtension.require_functions(
            ShadowBoundaryKernel._REQUIRED,
            context="Shared shadow-boundary native helper",
        )

    @staticmethod
    def available() -> bool:
        return NativeExtension.has_functions(ShadowBoundaryKernel._REQUIRED)

    @staticmethod
    def accumulate(
        *,
        states,
        grid,
        k: float,
        wavelength: float,
        tile_shape: tuple[int, int],
        band_width_wavelengths: float,
        max_candidate_factor: float,
        direct_los_visible,
        direct_blocker_group,
        edge_adjacent_group0,
        edge_adjacent_group1,
    ) -> dict[str, object]:
        n_states = int(dr.width(states.edge_pos.x))
        n_cells = int(grid.n_cells)
        if n_states <= 0 or n_cells <= 0:
            zero = dr.zeros(wt.Float, n_cells)
            return {
                "incident_shadow_boundary_weight": zero,
                "reflection_shadow_boundary_weight": zero,
                "incident_transition_response_real": zero,
                "incident_transition_response_imag": zero,
                "reflection_transition_response_real": zero,
                "reflection_transition_response_imag": zero,
                "_metadata": {
                    "backend": "native_candidate",
                    "candidate_tiles": 0,
                    "candidate_pairs": 0,
                    "candidate_ratio": 0.0,
                    "tile_shape": tuple(int(v) for v in tile_shape),
                    "band_width_wavelengths": float(band_width_wavelengths),
                    "weight_aggregation": "max_weight_weighted_response_average",
                    "incident_support": (
                        "direct_los_or_matching_first_blocker_surface_group"
                    ),
                },
            }

        ext = ShadowBoundaryKernel.require()
        tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
        dr.eval(
            states.edge_pos,
            states.edge_dir,
            states.n0,
            states.n_face_n,
            states.wedge_n,
            states.edge_line_min,
            states.edge_line_max,
            states.source_pos,
            direct_los_visible,
            direct_blocker_group,
            edge_adjacent_group0,
            edge_adjacent_group1,
            grid.cell_centers,
        )
        (
            incident_weight,
            reflection_weight,
            incident_real,
            incident_imag,
            reflection_real,
            reflection_imag,
            candidate_tile_count,
            candidate_cell_count,
        ) = getattr(ext, ShadowBoundaryKernel.FUNCTION_NAME)(
            states.edge_pos.x,
            states.edge_pos.y,
            states.edge_pos.z,
            states.edge_dir.x,
            states.edge_dir.y,
            states.edge_dir.z,
            states.n0.x,
            states.n0.y,
            states.n0.z,
            states.n_face_n.x,
            states.n_face_n.y,
            states.n_face_n.z,
            states.wedge_n,
            states.edge_line_min,
            states.edge_line_max,
            states.source_pos.x,
            states.source_pos.y,
            states.source_pos.z,
            wt.UInt32(direct_los_visible),
            wt.Int32(direct_blocker_group),
            wt.Int32(edge_adjacent_group0),
            wt.Int32(edge_adjacent_group1),
            grid.cell_centers.x,
            grid.cell_centers.y,
            grid.cell_centers.z,
            int(n_states),
            int(grid.grid_shape[0]),
            int(grid.grid_shape[1]),
            int(tile_shape[0]),
            int(tile_shape[1]),
            float(k),
            float(wavelength),
            float(band_width_wavelengths),
            float(max_candidate_factor),
        )
        dr.eval(candidate_tile_count, candidate_cell_count)
        candidate_tiles = int(candidate_tile_count[0])
        candidate_cells = int(candidate_cell_count[0])
        full_pair_count = int(n_states) * int(n_cells)
        candidate_pairs = min(candidate_cells, full_pair_count)
        full_pairs = max(1, full_pair_count)
        metadata = {
            "backend": "native_candidate",
            "candidate_tiles": candidate_tiles,
            "candidate_pairs": candidate_pairs,
            "candidate_ratio": float(candidate_pairs) / float(full_pairs),
            "tile_shape": tuple(tile_shape),
            "band_width_wavelengths": float(band_width_wavelengths),
            "weight_aggregation": "max_weight_weighted_response_average",
            "incident_support": "direct_los_or_matching_first_blocker_surface_group",
        }
        return {
            "incident_shadow_boundary_weight": wt.Float(incident_weight),
            "reflection_shadow_boundary_weight": wt.Float(reflection_weight),
            "incident_transition_response_real": wt.Float(incident_real),
            "incident_transition_response_imag": wt.Float(incident_imag),
            "reflection_transition_response_real": wt.Float(reflection_real),
            "reflection_transition_response_imag": wt.Float(reflection_imag),
            "_metadata": metadata,
        }


__all__ = ["ShadowBoundaryKernel"]
