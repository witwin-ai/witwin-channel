# Copyright Xingyu Chen.
# Benchmarks run maxwell single cube.

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(
    os.environ.get(
        "WITWIN_FULLWAVE_OUTPUT_DIR",
        ROOT / "artifacts/fullwave/single-cube-metal-z042",
    )
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
maxwell_source = os.environ.get("WITWIN_MAXWELL_SOURCE")
if maxwell_source:
    sys.path.insert(0, maxwell_source)
sys.path.insert(0, str(ROOT))

# witwin-maxwell currently selects gcc-10 when its FDTD module is imported.
# Restore the active Visual Studio compiler for this Windows smoke run.
os.environ["CC"] = "cl"
os.environ["CXX"] = "cl"

import maxwell as mw  # noqa: E402
import maxwell.fdtd.solver as _fdtd_solver  # noqa: E402, F401

from benchmarks.fullwave_validation.backends import solve_deterministic  # noqa: E402
from benchmarks.fullwave_validation.models import FieldMap, MaterialSpec  # noqa: E402
from benchmarks.fullwave_validation.scenarios import load_case  # noqa: E402


spec = replace(
    load_case("single_cube", "metal"),
    material=MaterialSpec(kind="metal", eps_r=1.0, sigma_e=0.0),
    cube_centers=((0.0, 0.0, 0.15),),
    cube_size_m=0.2,
)
deterministic = solve_deterministic(spec)
deterministic.save(
    OUTPUT_DIR / "visual-deterministic-metal-centered-5ghz-256.npz"
)

scene = mw.Scene(
    domain=mw.Domain(bounds=spec.domain_bounds_xyz),
    grid=mw.GridSpec.uniform(spec.fullwave_dl_m),
    boundary=mw.BoundarySpec.pml(
        num_layers=spec.fullwave_pml_layers,
        strength=1.0,
    ),
    device="cuda",
)
for index, center in enumerate(spec.cube_centers, start=1):
    scene.with_structure(
        mw.Structure(
            geometry=mw.Box(center=center, size=(spec.cube_size_m,) * 3),
            # witwin-maxwell does not expose a PEC *volume* medium. Infinite
            # permittivity makes the electric update coefficient exactly zero
            # inside the voxelized cube, enforcing zero interior E/no transmission.
            medium=mw.Medium(eps_r=float("inf"), name="pec-volume"),
            name=f"cube-{index}",
        )
    )
scene.with_source(
    mw.PointDipole(
        center=spec.tx_position,
        polarization="Ez",
        profile="ideal",
        source_time=mw.GaussianPulse(
            frequency=spec.frequency_hz,
            fwidth=spec.frequency_hz / 5.0,
        ),
        name="tx",
    )
)
scene.with_monitor(
    mw.PlaneMonitor(
        "field_xy",
        axis="z",
        position=spec.plane_z,
        fields=("Ez",),
        freqs=(spec.frequency_hz,),
    )
)
material_model = scene.get_compiled_material_model()
ix = int(torch.argmin(torch.abs(scene.x - spec.cube_centers[0][0])).item())
iy = int(torch.argmin(torch.abs(scene.y - spec.cube_centers[0][1])).item())
iz = int(torch.argmin(torch.abs(scene.z - spec.cube_centers[0][2])).item())
if not bool(torch.isinf(material_model["eps_r"][ix, iy, iz]).item()):
    raise RuntimeError("PEC cube center was not voxelized with infinite permittivity")
result = mw.Simulation.fdtd(
    scene,
    freqs=(spec.frequency_hz,),
    run_time=mw.RunTime.auto(steady_cycles=12, transient_cycles=12),
    spectral_sampler=mw.DFT(window="hanning", normalize_source=True),
    full_field_dft=False,
).run()
monitor = result.monitor("field_xy")
values = monitor["data"]
if isinstance(values, torch.Tensor):
    values = values.detach().cpu().numpy()
monitor_x = np.asarray(monitor["x"], dtype=np.float64)
monitor_y = np.asarray(monitor["y"], dtype=np.float64)
field_yx = np.asarray(values, dtype=np.complex128).T
x_indices = np.asarray([np.argmin(np.abs(monitor_x - value)) for value in spec.x])
y_indices = np.asarray([np.argmin(np.abs(monitor_y - value)) for value in spec.y])
alignment_error_m = max(
    float(np.max(np.abs(monitor_x[x_indices] - spec.x))),
    float(np.max(np.abs(monitor_y[y_indices] - spec.y))),
)
if alignment_error_m > 1.0e-7:
    raise RuntimeError(
        f"Maxwell and deterministic sample grids are misaligned by {alignment_error_m:.3e} m"
    )
plane_positions = tuple(float(value) for value in monitor.get("plane_sample_positions", ()))
if plane_positions and min(abs(value - spec.plane_z) for value in plane_positions) > 1.0e-7:
    raise RuntimeError(
        f"Maxwell Ez plane does not include z={spec.plane_z}: {plane_positions}"
    )
strict_interior = (
    (np.abs(monitor_y[:, None]) <= 0.07)
    & (np.abs(monitor_x[None, :]) <= 0.07)
)
interior_peak = float(np.max(np.abs(field_yx[strict_interior])))
exterior_peak = float(np.max(np.abs(field_yx[~strict_interior])))
interior_to_exterior = interior_peak / max(exterior_peak, 1.0e-30)
if interior_to_exterior > 1.0e-7:
    raise RuntimeError(
        f"PEC interior field is not zero: peak ratio={interior_to_exterior:.3e}"
    )
fullwave = FieldMap(
    x=monitor_x,
    y=monitor_y,
    field=field_yx,
    metadata={
        "backend": "witwin-maxwell-fdtd",
        "case_id": spec.case_id,
        "case_fingerprint": spec.fingerprint,
        "frequency_hz": spec.frequency_hz,
        "field_component": "Ez",
        "material": "pec-volume",
        "pec_implementation": "eps_r=inf; zero electric update coefficient",
        "pec_interior_to_exterior_peak_ratio": interior_to_exterior,
        "sample_alignment_error_m": alignment_error_m,
        "plane_sample_positions_m": plane_positions,
        "time_steps": result.stats()["time_steps"],
    },
)
fullwave.save(OUTPUT_DIR / "visual-maxwell-metal-centered-5ghz-256.npz")
print(result.stats())