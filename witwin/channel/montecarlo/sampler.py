from __future__ import annotations

import math
from typing import Mapping

import drjit as dr
from witwin.channel.montecarlo import types as wt
from witwin.channel.core.numerics.arrays import scalar
from witwin.channel.core.numerics.constants import EPS
from witwin.channel.core.physics import polarization
_MC_UINT32_MASK = 0xFFFFFFFF
_SOBOL_INV_24 = 1.0 / 16777216.0
_SOBOL_DIMENSIONS = (
    (1, 0, (1,)),
    (2, 1, (1, 3)),
    (3, 1, (1, 3, 1)),
    (3, 2, (1, 1, 1)),
    (4, 1, (1, 3, 5, 13)),
    (4, 4, (1, 1, 5, 5)),
    (5, 2, (1, 1, 5, 5, 17)),
    (5, 4, (1, 1, 5, 5, 5)),
    (5, 7, (1, 1, 7, 11, 19)),
    (5, 11, (1, 1, 5, 1, 1)),
    (5, 13, (1, 1, 1, 3, 11)),
    (5, 14, (1, 3, 5, 5, 31)),
    (6, 1, (1, 3, 3, 9, 7, 49)),
    (6, 13, (1, 1, 1, 15, 21, 21)),
    (6, 16, (1, 3, 1, 13, 27, 49)),
)


class Sampler:
    """Ray sampling and source field construction."""

    @staticmethod
    def _hash_uniform_bits(index, *, stream: int, seed: int):
        resolved_seed = int(seed) & _MC_UINT32_MASK
        stream_value = wt.UInt32(stream) + wt.UInt32(1)
        value = (
            index * wt.UInt32(747796405)
            + wt.UInt32(resolved_seed + 1) * wt.UInt32(2891336453)
            + stream_value * wt.UInt32(277803737)
        )
        value = (value ^ (value >> 16)) * wt.UInt32(2246822519)
        value = (value ^ (value >> 13)) * wt.UInt32(3266489917)
        return value ^ (value >> 16)

    @staticmethod
    def _sobol_direction_numbers(dimension: int) -> tuple[int, ...]:
        if int(dimension) < 0:
            raise ValueError("Sobol dimension must be non-negative.")
        if int(dimension) == 0:
            return tuple(1 << (31 - bit) for bit in range(32))
        table_index = int(dimension) - 1
        if table_index >= len(_SOBOL_DIMENSIONS):
            raise ValueError(
                "BDPT Sobol diffraction sampling currently supports dimensions "
                f"0..{len(_SOBOL_DIMENSIONS)}."
            )
        degree, coefficient, initial_values = _SOBOL_DIMENSIONS[table_index]
        directions = [int(value) << (32 - bit - 1) for bit, value in enumerate(initial_values)]
        for bit in range(int(degree), 32):
            value = directions[bit - int(degree)] ^ (
                directions[bit - int(degree)] >> int(degree)
            )
            for coeff_bit in range(1, int(degree)):
                if (int(coefficient) >> (int(degree) - 1 - coeff_bit)) & 1:
                    value ^= directions[bit - coeff_bit]
            directions.append(value)
        return tuple(directions)

    @staticmethod
    def axis_unit_normal(axis: str):
        if axis == "x":
            return wt.Vector3f(1.0, 0.0, 0.0)
        if axis == "y":
            return wt.Vector3f(0.0, 1.0, 0.0)
        return wt.Vector3f(0.0, 0.0, 1.0)

    @staticmethod
    def spawn_offset_ray_origin(point_pos, ray_dir, normal_dir):
        normal_hat = normal_dir / (dr.norm(normal_dir) + wt.Float(EPS))
        direction_sign = dr.sign(dr.dot(ray_dir, normal_hat))
        offset_scale = wt.Float(1.0e-5) * (
            wt.Float(1.0) + dr.max(dr.abs(point_pos), axis=0)
        )
        signed_offset = dr.detach(dr.mulsign(offset_scale, direction_sign))
        return point_pos + signed_offset * normal_hat

    @staticmethod
    def directions(n_rays: int, *, ray_index=None):
        if ray_index is None:
            if n_rays <= 0:
                zero = dr.zeros(wt.Float, 0)
                return wt.Vector3f(zero, zero, zero)
            width = int(n_rays)
        else:
            width = int(dr.width(ray_index))
            if n_rays <= 0 or width <= 0:
                zero = dr.zeros(wt.Float, width)
                return wt.Vector3f(zero, zero, zero)
        float64_t = dr.float64_array_t(wt.Float)
        if ray_index is None:
            indices = dr.arange(float64_t, 0, n_rays)
        else:
            indices = float64_t(ray_index)
        golden_ratio = float64_t((1.0 + math.sqrt(5.0)) / 2.0)
        azimuth_u = indices / golden_ratio
        azimuth_u = azimuth_u - dr.floor(azimuth_u)
        if n_rays == 1:
            elevation_v = dr.zeros(float64_t, width)
        else:
            elevation_v = indices / float64_t(n_rays - 1)
        phi = wt.Float(float(2.0 * math.pi)) * wt.Float(azimuth_u)
        z = wt.Float(1.0) - wt.Float(2.0) * wt.Float(elevation_v)
        radial = dr.sqrt(dr.maximum(wt.Float(0.0), wt.Float(1.0) - z * z))
        sin_phi, cos_phi = dr.sincos(phi)
        return wt.Vector3f(radial * cos_phi, radial * sin_phi, z)

    @staticmethod
    def metadata(*, axis: str, plane_position: float, tx_pos):
        plane_distance = abs(float(plane_position) - float(scalar(getattr(tx_pos, str(axis)))))
        return {
            "requested_ray_sampling": "full_sphere",
            "selected_ray_sampling": "full_sphere",
            "sampling_sequence": "sionna_fibonacci_square_to_uniform_sphere",
            "monitor_plane_distance_to_tx": plane_distance,
            "near_plane_sampling_threshold": 0.0,
        }

    @staticmethod
    def solid_angle(ray_sampling_metadata: Mapping[str, object], samples_per_tx: int) -> float:
        if samples_per_tx <= 0:
            return 0.0
        selected = str(ray_sampling_metadata.get("selected_ray_sampling", "full_sphere"))
        if selected == "full_sphere":
            return float(4.0 * math.pi / samples_per_tx)
        if selected in {"hemisphere_facing_monitor", "circle_2d"}:
            return float(2.0 * math.pi / samples_per_tx)
        raise RuntimeError(f"Unsupported Monte Carlo ray distribution: {selected!r}")

    @staticmethod
    def source_field(ray_dir):
        return polarization.vector_from_scalar(
            wt.Complex2f(1.0, 0.0),
            polarization.implicit_basis_vector(ray_dir),
        )

    @staticmethod
    def hash_uniform(index, *, stream: int, seed: int):
        mantissa = Sampler._hash_uniform_bits(
            index,
            stream=int(stream),
            seed=int(seed),
        ) & wt.UInt32(0x00FFFFFF)
        return wt.Float(mantissa) / wt.Float(16777216.0)

    @staticmethod
    def sobol_uniform(index, *, dimension: int, seed: int):
        directions = Sampler._sobol_direction_numbers(int(dimension))
        sobol_index = wt.UInt32(index) + wt.UInt32(1)
        value = wt.UInt32(0)
        for bit, direction in enumerate(directions):
            bit_set = (sobol_index & wt.UInt32(1 << bit)) != wt.UInt32(0)
            value = dr.select(bit_set, value ^ wt.UInt32(direction), value)
        scramble = Sampler._hash_uniform_bits(
            wt.UInt32(int(dimension)),
            stream=1009,
            seed=int(seed),
        )
        value = value ^ scramble
        mantissa = (value >> 8) & wt.UInt32(0x00FFFFFF)
        return (wt.Float(mantissa) + wt.Float(0.5)) * wt.Float(_SOBOL_INV_24)


__all__ = [
    "Sampler",
]
