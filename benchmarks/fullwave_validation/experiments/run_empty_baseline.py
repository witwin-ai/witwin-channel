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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
maxwell_source = os.environ.get("WITWIN_MAXWELL_SOURCE")
if maxwell_source:
    sys.path.insert(0, maxwell_source)

# witwin-maxwell selects gcc-10 on FDTD import; use the active MSVC toolchain.
os.environ["CC"] = "cl"
os.environ["CXX"] = "cl"

import maxwell as mw  # noqa: E402
import maxwell.fdtd.solver as _fdtd_solver  # noqa: E402, F401

from benchmarks.fullwave_validation.models import FieldMap, MaterialSpec  # noqa: E402
from benchmarks.fullwave_validation.scenarios import (  # noqa: E402
    build_channel_scene,
    load_case,
)
from witwin.core import Scene  # noqa: E402
from witwin.channel.deterministic import Config, solve  # noqa: E402


spec = replace(
    load_case("single_cube", "metal"),
    material=MaterialSpec(kind="metal", eps_r=1.0, sigma_e=0.0),
    cube_centers=((0.0, 0.0, 0.15),),
    cube_size_m=0.2,
)

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
).save(OUTPUT_DIR / "visual-deterministic-empty-5ghz-256.npz")

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
FieldMap(
    x=np.asarray(monitor["x"], dtype=np.float64),
    y=np.asarray(monitor["y"], dtype=np.float64),
    field=np.asarray(values, dtype=np.complex128).T,
    metadata={
        "backend": "witwin-maxwell-fdtd-empty",
        "case_id": spec.case_id,
        "case_fingerprint": spec.fingerprint,
        "frequency_hz": spec.frequency_hz,
        "field_component": "Ez",
        "source_normalization": "source-spectrum normalized",
        "time_steps": maxwell_result.stats()["time_steps"],
    },
).save(OUTPUT_DIR / "visual-maxwell-empty-5ghz-256.npz")
print(maxwell_result.stats())
