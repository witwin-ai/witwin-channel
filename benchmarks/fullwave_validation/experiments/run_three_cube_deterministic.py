# Copyright Xingyu Chen.
# Three-cube deterministic solves for the Yee-locked ``three_cube_320`` case.

"""Three-cube deterministic solves for the Yee-locked ``three_cube_320`` case.

Runs the deterministic solver twice on the versioned ``three_cube_320`` /
``metal`` case: once with the coupled reflection-diffraction compensator
(ADR-011) disabled and once enabled, matching the benchmark backend
configuration in every other respect. Saves a FieldMap plus the full exported
path table for each run so boundary metrics can be decomposed per path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

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

from benchmarks.fullwave_validation.models import FieldMap  # noqa: E402
from benchmarks.fullwave_validation.scenarios import (  # noqa: E402
    build_channel_scene,
    load_case,
)
from witwin.channel.deterministic import Config, solve  # noqa: E402


COMPONENTS = frozenset({"los", "reflection", "diffraction", "transmission"})

spec = load_case("three_cube_320", "metal")
print(f"case {spec.case_id} fingerprint {spec.fingerprint}")


def _npy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _export_path_table(paths, path: Path) -> None:
    np.savez_compressed(
        path,
        valid=_npy(paths.valid),
        tx_id=_npy(paths.tx_id),
        rx_id=_npy(paths.rx_id),
        depth=_npy(paths.depth),
        component_id=_npy(paths.component_id),
        path_gain=_npy(paths.path_gain),
        interaction_positions=_npy(paths.interaction_positions),
        field_real=_npy(paths.field_real),
        field_imag=_npy(paths.field_imag),
    )


def run(coupled: bool) -> dict[str, object]:
    config = Config(
        components=COMPONENTS,
        max_depth=spec.max_depth,
        max_diffraction_order=1,
        coherent=True,
        return_field=True,
        export_paths=True,
        diagnostics=True,
        coupled_paths=coupled,
        coupled_candidate_limit=1_000_000,
    )
    result = solve(build_channel_scene(spec), config)
    torch.cuda.synchronize()

    scene = build_channel_scene(spec)
    torch.cuda.synchronize()
    start = time.perf_counter()
    warm = solve(scene, config)
    _ = warm.field.detach()
    torch.cuda.synchronize()
    warm_seconds = time.perf_counter() - start

    field = result.field.detach().cpu().numpy()
    if field.shape != (1, spec.y.size, spec.x.size):
        raise RuntimeError(f"unexpected deterministic field shape: {field.shape}")
    components = {
        name: values.detach().cpu().numpy()[0]
        for name, values in result.component_fields.items()
        if values.numel() > 0
    }
    suffix = "on" if coupled else "off"
    FieldMap(
        x=spec.x,
        y=spec.y,
        field=field[0],
        components=components,
        metadata={
            "backend": "deterministic",
            "case_id": spec.case_id,
            "case_fingerprint": spec.fingerprint,
            "frequency_hz": spec.frequency_hz,
            "components": sorted(COMPONENTS),
            "max_depth": spec.max_depth,
            "coupled_paths": coupled,
            "path_count": int(result.metadata["counts"]["path_count"]),
        },
    ).save(OUTPUT_DIR / f"three_cube_320_coupled_{suffix}.npz")
    _export_path_table(
        result.paths, OUTPUT_DIR / f"three_cube_320_pathtable_{suffix}.npz"
    )

    component_ids = _npy(result.paths.component_id)
    valid = _npy(result.paths.valid).astype(bool)
    summary = {
        "coupled_paths": coupled,
        "warm_seconds": warm_seconds,
        "components": sorted(components),
        "path_count": int(result.metadata["counts"]["path_count"]),
        "valid_rows": int(valid.sum()),
        "coupled_rows": int(np.isin(component_ids[valid], (3, 4)).sum()),
        "field_abs_max": float(np.abs(field[0]).max()),
        "coupled_paths_metadata": result.metadata.get("coupled_paths"),
    }
    print(json.dumps(summary, indent=2))
    return summary


summary = {"coupled_off": run(False), "coupled_on": run(True)}
(OUTPUT_DIR / "three_cube_320_solve_summary.json").write_text(
    json.dumps(summary, indent=2)
)