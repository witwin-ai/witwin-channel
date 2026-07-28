"""Empty-scene calibration pair for the ``three_cube_320`` case.

Runs the deterministic LoS-only solve and the witwin-maxwell FDTD on the same
empty domain/grid as the three-cube reference. The pair yields the frozen
positive amplitude scale ``s_empty = sqrt(sum|Ez_empty|^2 / sum|h_empty|^2)``
used by the metal comparison without refitting.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(
    os.environ.get(
        "WITWIN_FULLWAVE_OUTPUT_DIR",
        ROOT / "artifacts/fullwave/three-cube-metal-320",
    )
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
maxwell_source = os.environ.get("WITWIN_MAXWELL_SOURCE")
if maxwell_source:
    sys.path.insert(0, maxwell_source)

# witwin-maxwell selects gcc-10 on FDTD import; use the active MSVC toolchain.
os.environ["CC"] = "cl"
os.environ["CXX"] = "cl"

import maxwell as mw  # noqa: E402
import maxwell.fdtd.solver as _fdtd_solver  # noqa: E402, F401

from benchmarks.fullwave_validation.models import FieldMap  # noqa: E402
from benchmarks.fullwave_validation.scenarios import (  # noqa: E402
    build_channel_scene,
    load_case,
)
from witwin.core import Scene  # noqa: E402
from witwin.channel.deterministic import Config, solve  # noqa: E402


spec = load_case("three_cube_320", "metal")
print(f"case {spec.case_id} fingerprint {spec.fingerprint}")

channel_scene = build_channel_scene(spec)
empty_channel_scene = Scene(
    structures=[],
    transmitters=channel_scene.transmitters,
    receivers=channel_scene.receivers,
    frequency=channel_scene.frequency,
    metadata=channel_scene.metadata,
)
deterministic_result = solve(
    empty_channel_scene,
    Config(
        components=frozenset({"los"}),
        max_depth=0,
        coherent=True,
        return_field=True,
        export_paths=False,
        diagnostics=True,
    ),
)
deterministic_field = deterministic_result.field.detach().cpu().numpy()[0]
FieldMap(
    x=spec.x,
    y=spec.y,
    field=deterministic_field,
    metadata={
        "backend": "deterministic-empty",
        "case_id": spec.case_id,
        "case_fingerprint": spec.fingerprint,
        "frequency_hz": spec.frequency_hz,
    },
).save(OUTPUT_DIR / "visual-deterministic-empty-three-cube-5ghz-320.npz")

maxwell_scene = mw.Scene(
    domain=mw.Domain(bounds=spec.domain_bounds_xyz),
    grid=mw.GridSpec.uniform(spec.fullwave_dl_m),
    boundary=mw.BoundarySpec.pml(
        num_layers=spec.fullwave_pml_layers,
        strength=1.0,
    ),
    device="cuda",
)
maxwell_scene.with_source(
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
maxwell_scene.with_monitor(
    mw.PlaneMonitor(
        "field_xy",
        axis="z",
        position=spec.plane_z,
        fields=("Ez",),
        freqs=(spec.frequency_hz,),
    )
)
maxwell_result = mw.Simulation.fdtd(
    maxwell_scene,
    freqs=(spec.frequency_hz,),
    run_time=mw.RunTime.auto(steady_cycles=12, transient_cycles=12),
    spectral_sampler=mw.DFT(window="hanning", normalize_source=True),
    full_field_dft=False,
).run()
monitor = maxwell_result.monitor("field_xy")
values = monitor["data"]
if isinstance(values, torch.Tensor):
    values = values.detach().cpu().numpy()
monitor_x = np.asarray(monitor["x"], dtype=np.float64)
monitor_y = np.asarray(monitor["y"], dtype=np.float64)
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
FieldMap(
    x=monitor_x,
    y=monitor_y,
    field=np.asarray(values, dtype=np.complex128).T,
    metadata={
        "backend": "witwin-maxwell-fdtd-empty",
        "case_id": spec.case_id,
        "case_fingerprint": spec.fingerprint,
        "frequency_hz": spec.frequency_hz,
        "field_component": "Ez",
        "source_normalization": "source-spectrum normalized",
        "sample_alignment_error_m": alignment_error_m,
        "time_steps": maxwell_result.stats()["time_steps"],
    },
).save(OUTPUT_DIR / "visual-maxwell-empty-three-cube-5ghz-320.npz")
print(maxwell_result.stats())
