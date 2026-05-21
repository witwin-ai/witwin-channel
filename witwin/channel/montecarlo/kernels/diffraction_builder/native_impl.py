from __future__ import annotations

import drjit as dr
from witwin.channel.montecarlo import types as wt
from witwin.channel._native.montecarlo import NativeExtension
_REQUIRED = (
    "monte_carlo_diffraction_sample_slots",
    "monte_carlo_diffraction_best_edge_indices",
    "monte_carlo_diffraction_discover_edges",
    "monte_carlo_diffraction_build_state_arrays",
)


class DiffractionBuilderKernel:
    """Static namespace for native diffraction state-builder helpers."""

    @staticmethod
    def require():
        return NativeExtension.require_functions(_REQUIRED, context="Monte Carlo diffraction builder native helper")

    @staticmethod
    def available() -> bool:
        return NativeExtension.has_functions(_REQUIRED)

    @staticmethod
    def sample_slots(*, sample_index, cdf, n_states: int, total_length_scalar: float, seed: int):
        if int(dr.width(sample_index)) <= 0 or int(n_states) <= 0:
            return dr.zeros(wt.UInt32, 0)
        ext = DiffractionBuilderKernel.require()
        dr.eval(sample_index, cdf)
        return wt.UInt32(
            ext.monte_carlo_diffraction_sample_slots(
                sample_index,
                cdf,
                int(dr.width(sample_index)),
                int(n_states),
                float(total_length_scalar),
                int(seed),
            )
        )

    @staticmethod
    def best_edge_indices(
        *,
        tx_pos,
        ray_directions,
        hit_p,
        hit_n,
        hit_geo_n,
        hit,
        triangle_edge_count,
        triangle_edge_indices,
        max_triangle_edge_slots: int,
        edge_runtime,
    ):
        n_rays = int(dr.width(ray_directions.x))
        if n_rays <= 0:
            return dr.zeros(wt.Int32, 0)
        ext = DiffractionBuilderKernel.require()
        hit_i = wt.Int32(dr.select(hit, wt.Int32(1), wt.Int32(0)))
        dr.eval(
            ray_directions,
            hit_p,
            hit_n,
            hit_geo_n,
            hit_i,
            triangle_edge_count,
            triangle_edge_indices,
            edge_runtime["pos"],
            edge_runtime["edge_dir"],
            edge_runtime["n0"],
            edge_runtime["n_face_n"],
            edge_runtime["line_min"],
            edge_runtime["line_max"],
            edge_runtime["adjacent_face1"],
        )
        return wt.Int32(
            ext.monte_carlo_diffraction_best_edge_indices(
                float(tx_pos.x[0]),
                float(tx_pos.y[0]),
                float(tx_pos.z[0]),
                ray_directions.x,
                ray_directions.y,
                ray_directions.z,
                hit_p.x,
                hit_p.y,
                hit_p.z,
                hit_n.x,
                hit_n.y,
                hit_n.z,
                hit_geo_n.x,
                hit_geo_n.y,
                hit_geo_n.z,
                hit_i,
                triangle_edge_count,
                triangle_edge_indices,
                int(max_triangle_edge_slots),
                edge_runtime["pos"].x,
                edge_runtime["pos"].y,
                edge_runtime["pos"].z,
                edge_runtime["edge_dir"].x,
                edge_runtime["edge_dir"].y,
                edge_runtime["edge_dir"].z,
                edge_runtime["n0"].x,
                edge_runtime["n0"].y,
                edge_runtime["n0"].z,
                edge_runtime["n_face_n"].x,
                edge_runtime["n_face_n"].y,
                edge_runtime["n_face_n"].z,
                edge_runtime["line_min"],
                edge_runtime["line_max"],
                edge_runtime["adjacent_face1"],
                int(dr.width(edge_runtime["line_min"])),
                n_rays,
            )
        )

    @staticmethod
    def discover_edge_indices_from_hits(
        *,
        tx_pos,
        ray_directions,
        prim_index,
        hit_p,
        hit_n,
        hit_geo_n,
        n_hits: int,
        triangle_edge_count,
        triangle_edge_indices,
        max_triangle_edge_slots: int,
        n_triangles: int,
        edge_runtime,
    ):
        if int(n_hits) <= 0:
            return dr.zeros(wt.UInt32, 0)
        ext = DiffractionBuilderKernel.require()
        dr.eval(
            ray_directions,
            prim_index,
            hit_p,
            hit_n,
            hit_geo_n,
            triangle_edge_count,
            triangle_edge_indices,
            edge_runtime["pos"],
            edge_runtime["edge_dir"],
            edge_runtime["n0"],
            edge_runtime["n_face_n"],
            edge_runtime["line_min"],
            edge_runtime["line_max"],
            edge_runtime["adjacent_face1"],
        )
        return wt.UInt32(
            ext.monte_carlo_diffraction_discover_edges(
                float(tx_pos.x[0]),
                float(tx_pos.y[0]),
                float(tx_pos.z[0]),
                ray_directions.x,
                ray_directions.y,
                ray_directions.z,
                prim_index,
                hit_p.x,
                hit_p.y,
                hit_p.z,
                hit_n.x,
                hit_n.y,
                hit_n.z,
                hit_geo_n.x,
                hit_geo_n.y,
                hit_geo_n.z,
                int(n_hits),
                triangle_edge_count,
                triangle_edge_indices,
                int(max_triangle_edge_slots),
                int(n_triangles),
                edge_runtime["pos"].x,
                edge_runtime["pos"].y,
                edge_runtime["pos"].z,
                edge_runtime["edge_dir"].x,
                edge_runtime["edge_dir"].y,
                edge_runtime["edge_dir"].z,
                edge_runtime["n0"].x,
                edge_runtime["n0"].y,
                edge_runtime["n0"].z,
                edge_runtime["n_face_n"].x,
                edge_runtime["n_face_n"].y,
                edge_runtime["n_face_n"].z,
                edge_runtime["line_min"],
                edge_runtime["line_max"],
                edge_runtime["adjacent_face1"],
                int(dr.width(edge_runtime["line_min"])),
            )
        )

    @staticmethod
    def build_state_arrays(*, edge_idx, tx_pos, edge_runtime):
        n_states = int(dr.width(edge_idx))
        if n_states <= 0:
            return None
        ext = DiffractionBuilderKernel.require()
        dr.eval(
            edge_idx,
            edge_runtime["pos"],
            edge_runtime["edge_dir"],
            edge_runtime["n0"],
            edge_runtime["n_face_n"],
            edge_runtime["wedge_n"],
            edge_runtime["line_min"],
            edge_runtime["line_max"],
            edge_runtime["adjacent_face0"],
            edge_runtime["adjacent_face1"],
        )
        payload = ext.monte_carlo_diffraction_build_state_arrays(
            edge_idx,
            n_states,
            float(tx_pos.x[0]),
            float(tx_pos.y[0]),
            float(tx_pos.z[0]),
            edge_runtime["pos"].x,
            edge_runtime["pos"].y,
            edge_runtime["pos"].z,
            edge_runtime["edge_dir"].x,
            edge_runtime["edge_dir"].y,
            edge_runtime["edge_dir"].z,
            edge_runtime["n0"].x,
            edge_runtime["n0"].y,
            edge_runtime["n0"].z,
            edge_runtime["n_face_n"].x,
            edge_runtime["n_face_n"].y,
            edge_runtime["n_face_n"].z,
            edge_runtime["wedge_n"],
            edge_runtime["line_min"],
            edge_runtime["line_max"],
            edge_runtime["adjacent_face0"],
            edge_runtime["adjacent_face1"],
        )
        return {
            "edge_index": wt.Int32(payload["edge_index"]),
            "edge_pos": wt.Point3f(payload["edge_pos_x"], payload["edge_pos_y"], payload["edge_pos_z"]),
            "edge_dir": wt.Vector3f(payload["edge_dir_x"], payload["edge_dir_y"], payload["edge_dir_z"]),
            "n0": wt.Vector3f(payload["n0_x"], payload["n0_y"], payload["n0_z"]),
            "n_face_n": wt.Vector3f(payload["nn_x"], payload["nn_y"], payload["nn_z"]),
            "wedge_n": wt.Float(payload["wedge_n"]),
            "line_min": wt.Float(payload["line_min"]),
            "line_max": wt.Float(payload["line_max"]),
            "source_pos": wt.Point3f(payload["source_pos_x"], payload["source_pos_y"], payload["source_pos_z"]),
            "adjacent_face0": wt.Int32(payload["adjacent_face0"]),
            "adjacent_face1": wt.Int32(payload["adjacent_face1"]),
        }


__all__ = [
    "DiffractionBuilderKernel",
]
