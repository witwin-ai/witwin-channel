from __future__ import annotations

import drjit as dr

import witwin as wt
from witwin.channel._native import _extension, extension_available


def _require_radio_map_monte_carlo_kernel():
    ext = _extension()
    required = ("radiomap_monte_carlo_scatter_axis_aligned_into",)
    missing = [name for name in required if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(
            "Native radio-map Monte Carlo kernel requires "
            + ", ".join(missing)
            + ". Rebuild the witwin.channel native extension."
        )
    return ext


def radio_map_monte_carlo_kernel_available() -> bool:
    if not extension_available():
        return False
    ext = _extension()
    return hasattr(ext, "radiomap_monte_carlo_scatter_axis_aligned_into")


def scatter_axis_aligned_monte_carlo_samples(
    *,
    coord_0,
    coord_1,
    los_power,
    reflection_power,
    diffraction_power,
    out_los,
    out_reflection,
    out_diffraction,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    cell_size: tuple[float, float],
    grid_shape: tuple[int, int],
) -> None:
    n_samples = int(dr.width(coord_0))
    if n_samples <= 0:
        return
    ext = _require_radio_map_monte_carlo_kernel()
    dr.eval(
        coord_0,
        coord_1,
        los_power,
        reflection_power,
        diffraction_power,
        out_los,
        out_reflection,
        out_diffraction,
    )
    ext.radiomap_monte_carlo_scatter_axis_aligned_into(
        coord_0,
        coord_1,
        los_power,
        reflection_power,
        diffraction_power,
        out_los,
        out_reflection,
        out_diffraction,
        n_samples,
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
        float(cell_size[0]),
        float(cell_size[1]),
        int(grid_shape[0]),
        int(grid_shape[1]),
    )
    dr.make_opaque(out_los, out_reflection, out_diffraction)


__all__ = [
    "radio_map_monte_carlo_kernel_available",
    "scatter_axis_aligned_monte_carlo_samples",
]
