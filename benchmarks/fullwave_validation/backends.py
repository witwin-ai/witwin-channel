# Copyright Xingyu Chen.
# Benchmarks backends.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import CaseSpec, FieldMap
from .scenarios import build_channel_scene


_COMPONENTS = frozenset({"los", "reflection", "diffraction", "transmission"})
_METERS_TO_MICROMETERS = 1.0e6


def solve_deterministic(spec: CaseSpec) -> FieldMap:
    from witwin.channel.deterministic import Config, solve

    result = solve(
        build_channel_scene(spec),
        Config(
            components=_COMPONENTS,
            max_depth=spec.max_depth,
            max_diffraction_order=1,
            coherent=True,
            return_field=True,
            export_paths=True,
            diagnostics=True,
            # Full physics: the coupled reflection-diffraction compensator
            # (coupled reflection and diffraction). "coupled" is not a Config component; it is enabled by
            # this gate and exported as its own component map. Requires
            # max_depth >= 2 plus reflection + diffraction (both in
            # _COMPONENTS), so degenerate low-depth cases stay valid.
            coupled_paths=spec.max_depth >= 2,
            coupled_candidate_limit=1_000_000,
        ),
    )
    field = result.field.detach().cpu().numpy()
    if field.shape != (1, spec.y.size, spec.x.size):
        raise RuntimeError(f"unexpected deterministic field shape: {field.shape}")
    component_fields = {
        name: values.detach().cpu().numpy()[0]
        for name, values in result.component_fields.items()
        if values.numel() > 0
    }
    return FieldMap(
        x=spec.x,
        y=spec.y,
        field=field[0],
        components=component_fields,
        metadata={
            "backend": "deterministic",
            "case_id": spec.case_id,
            "case_fingerprint": spec.fingerprint,
            "frequency_hz": spec.frequency_hz,
            "components": sorted(_COMPONENTS),
            "max_depth": spec.max_depth,
            "path_count": int(result.metadata["counts"]["path_count"]),
        },
    )


def build_tidy3d_simulation(spec: CaseSpec) -> Any:
    try:
        import tidy3d as td
    except ImportError as exc:
        raise ImportError("Tidy3D is required for the full-wave backend") from exc

    scale = _METERS_TO_MICROMETERS
    structures = []
    medium = (
        td.PECMedium(name="pec")
        if spec.material.kind == "metal"
        else td.Medium(
            permittivity=spec.material.eps_r,
            conductivity=spec.material.sigma_e / scale,
            name="dielectric",
        )
    )
    for index, center in enumerate(spec.cube_centers, start=1):
        structures.append(
            td.Structure(
                geometry=td.Box(
                    center=tuple(value * scale for value in center),
                    size=(spec.cube_size_m * scale,) * 3,
                ),
                medium=medium,
                name=f"cube-{index}",
            )
        )

    domain_center = tuple((lo + hi) / 2.0 * scale for lo, hi in spec.domain_bounds_xyz)
    domain_size = tuple((hi - lo) * scale for lo, hi in spec.domain_bounds_xyz)
    (xmin, xmax), (ymin, ymax) = spec.analysis_bounds_xy
    monitor = td.FieldMonitor(
        center=(
            (xmin + xmax) / 2.0 * scale,
            (ymin + ymax) / 2.0 * scale,
            spec.plane_z * scale,
        ),
        size=((xmax - xmin) * scale, (ymax - ymin) * scale, 0.0),
        freqs=(spec.frequency_hz,),
        fields=("Ez",),
        colocate=True,
        name="field_xy",
    )
    source = td.PointDipole(
        center=tuple(value * scale for value in spec.tx_position),
        polarization="Ez",
        source_time=td.GaussianPulse(
            freq0=spec.frequency_hz,
            fwidth=spec.frequency_hz / 5.0,
        ),
        name="tx",
    )
    return td.Simulation(
        center=domain_center,
        size=domain_size,
        grid_spec=td.GridSpec.uniform(dl=spec.fullwave_dl_m * scale),
        structures=tuple(structures),
        sources=(source,),
        monitors=(monitor,),
        boundary_spec=td.BoundarySpec.all_sides(
            boundary=td.PML(num_layers=spec.fullwave_pml_layers)
        ),
        run_time=40.0 / spec.frequency_hz,
        normalize_index=0,
        shutoff=1.0e-6,
    )


def save_tidy3d_simulation(spec: CaseSpec, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_tidy3d_simulation(spec).to_file(str(output))
    return output


def extract_tidy3d_field_map(simulation_data: Any, spec: CaseSpec) -> FieldMap:
    monitor = simulation_data["field_xy"]
    values = monitor.Ez
    if "f" in values.dims:
        values = values.sel(f=spec.frequency_hz, method="nearest")
    for dim in tuple(values.dims):
        if dim not in {"x", "y"}:
            values = values.isel({dim: 0})
    values = values.transpose("y", "x")
    return FieldMap(
        x=np.asarray(values.coords["x"], dtype=np.float64) / _METERS_TO_MICROMETERS,
        y=np.asarray(values.coords["y"], dtype=np.float64) / _METERS_TO_MICROMETERS,
        field=np.asarray(values, dtype=np.complex128),
        metadata={
            "backend": "tidy3d",
            "case_id": spec.case_id,
            "case_fingerprint": spec.fingerprint,
            "frequency_hz": spec.frequency_hz,
            "field_component": "Ez",
            "source_normalization": "Tidy3D source-spectrum normalized",
            "tidy3d_length_unit": "micrometer",
            "reference_coordinate_unit": "meter",
        },
    )


def load_tidy3d_data(path: str | Path, spec: CaseSpec) -> FieldMap:
    try:
        import tidy3d as td
    except ImportError as exc:
        raise ImportError("Tidy3D is required to read SimulationData") from exc
    return extract_tidy3d_field_map(td.SimulationData.from_file(str(path)), spec)


def submit_tidy3d(spec: CaseSpec, *, task_name: str, data_path: str | Path) -> FieldMap:
    try:
        from tidy3d import web
    except ImportError as exc:
        raise ImportError("Tidy3D web client is required for cloud submission") from exc
    data = web.run(
        build_tidy3d_simulation(spec),
        task_name=task_name,
        path=str(data_path),
    )
    return extract_tidy3d_field_map(data, spec)