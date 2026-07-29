# Copyright Xingyu Chen.
# Coupled-OFF / coupled-ON versus FDTD comparison for ``three_cube_320``.

"""Coupled-OFF / coupled-ON versus FDTD comparison for ``three_cube_320``."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(
    os.environ.get(
        "WITWIN_FULLWAVE_OUTPUT_DIR",
        ROOT / "artifacts/fullwave/three-cube-metal-320",
    )
).resolve()
sys.path.insert(0, str(ROOT))

from benchmarks.fullwave_validation.metrics import (  # noqa: E402
    analyze_boundaries,
    compare_magnitudes,
    resample_regular,
)
from benchmarks.fullwave_validation.models import FieldMap  # noqa: E402
from benchmarks.fullwave_validation.scenarios import (  # noqa: E402
    load_case,
    observation_valid_mask,
)


FLAGSHIP_Y = 0.457


def coherence(candidate: FieldMap, reference: FieldMap, mask: np.ndarray) -> dict:
    """Complex coherence between deterministic h and Maxwell Ez over valid cells.

 The two engines use opposite time-sign conventions, which appears as
 complex conjugation; the reported coherence applies the conjugate
 convention and removes one global phase (|gamma|).
 """
    aligned = resample_regular(reference, candidate.x, candidate.y)
    c = np.conj(candidate.field[mask])
    r = aligned.field[mask]
    gamma = np.vdot(c, r) / np.sqrt(np.vdot(c, c).real * np.vdot(r, r).real)
    return {
        "coherence_after_global_phase": float(abs(gamma)),
        "complex_gamma": [float(gamma.real), float(gamma.imag)],
    }


def det_jumps(field: np.ndarray, mask: np.ndarray) -> dict:
    """Adjacent-cell |delta dB| over valid pairs of the deterministic map."""
    magnitude = np.abs(field)
    floor = max(float(magnitude.max()) * 1.0e-10, 1.0e-30)
    field_db = 20.0 * np.log10(np.maximum(magnitude, floor))
    horizontal_valid = mask[:, 1:] & mask[:, :-1]
    vertical_valid = mask[1:, :] & mask[:-1, :]
    jumps = np.concatenate(
        [
            np.abs(field_db[:, 1:] - field_db[:, :-1])[horizontal_valid],
            np.abs(field_db[1:, :] - field_db[:-1, :])[vertical_valid],
        ]
    )
    return {
        "median_db": float(np.median(jumps)),
        "p99_db": float(np.percentile(jumps, 99.0)),
        "max_db": float(jumps.max()),
        "n_pairs": int(jumps.size),
    }


def gap_map(candidate: FieldMap, reference: FieldMap, mask: np.ndarray, scale: float):
    """Per-cell magnitude gap 20*log10(s*|h| / |Ez|) over valid cells."""
    aligned = resample_regular(reference, candidate.x, candidate.y)
    floor = 1.0e-30
    gap_db = 20.0 * np.log10(
        np.maximum(scale * np.abs(candidate.field), floor)
        / np.maximum(np.abs(aligned.field), floor)
    )
    return np.where(mask, gap_db, np.nan)


def column(deterministic: FieldMap, fullwave: FieldMap, mask: np.ndarray, scale: float) -> dict:
    magnitude = compare_magnitudes(
        deterministic, fullwave, valid_mask=mask, amplitude_scale=scale
    )
    boundaries = analyze_boundaries(deterministic, fullwave, valid_mask=mask)
    result = {
        "envelope_nmse": float(magnitude.calibrated_magnitude_nmse),
        "magnitude_correlation": float(magnitude.magnitude_correlation),
        "magnitude_rmse_db": float(magnitude.magnitude_rmse_db),
        **coherence(deterministic, fullwave, mask),
        "det_only_jumps": det_jumps(deterministic.field, mask),
    }
    for name, metrics in boundaries.items():
        result[name.lower()] = {
            "p95_excess_jump_db": metrics.p95_excess_jump_db,
            "deterministic_jump_db_p95": metrics.deterministic_jump_db_p95,
            "fullwave_jump_db_p95": metrics.fullwave_jump_db_p95,
            "edge_count": metrics.edge_count,
        }
    return result


def flagship_line_scan(maps: dict[str, FieldMap], reference: FieldMap, scale: float) -> dict:
    """|field| dB along x at the flagship occlusion row (y ~ 0.457)."""
    any_map = next(iter(maps.values()))
    row = int(np.argmin(np.abs(any_map.y - FLAGSHIP_Y)))
    aligned = resample_regular(reference, any_map.x, any_map.y)
    floor = 1.0e-30

    def _db(values: np.ndarray, amplitude_scale: float) -> list[float]:
        return (
            20.0 * np.log10(np.maximum(amplitude_scale * np.abs(values), floor))
        ).tolist()

    scan = {
        "y_m": float(any_map.y[row]),
        "x_m": any_map.x.tolist(),
        "fullwave_db": _db(aligned.field[row], 1.0),
    }
    for name, field_map in maps.items():
        scan[f"{name}_db"] = _db(field_map.field[row], scale)
    return scan


spec = load_case("three_cube_320", "metal")
fullwave = FieldMap.load(OUTPUT_DIR / "visual-maxwell-metal-three-cube-5ghz-320.npz")
fullwave_empty = FieldMap.load(
    OUTPUT_DIR / "visual-maxwell-empty-three-cube-5ghz-320.npz"
)
deterministic_empty = FieldMap.load(
    OUTPUT_DIR / "visual-deterministic-empty-three-cube-5ghz-320.npz"
)
coupled_off = FieldMap.load(OUTPUT_DIR / "three_cube_320_coupled_off.npz")
coupled_on = FieldMap.load(OUTPUT_DIR / "three_cube_320_coupled_on.npz")

for candidate in (fullwave, fullwave_empty, coupled_off, coupled_on):
    if candidate.metadata["case_fingerprint"] != spec.fingerprint:
        raise ValueError(
            f"case fingerprint mismatch for {candidate.metadata['backend']}"
        )

mask = observation_valid_mask(spec, coupled_off.x, coupled_off.y)
s_empty = compare_magnitudes(
    deterministic_empty, fullwave_empty, valid_mask=mask
).amplitude_scale
empty_coherence = coherence(deterministic_empty, fullwave_empty, mask)

report = {
    "case_id": spec.case_id,
    "case_fingerprint": spec.fingerprint,
    "s_empty": float(s_empty),
    "empty_scene_coherence": empty_coherence,
    "coupled_off": column(coupled_off, fullwave, mask, s_empty),
    "coupled_on": column(coupled_on, fullwave, mask, s_empty),
    "flagship_line_scan": flagship_line_scan(
        {"coupled_off": coupled_off, "coupled_on": coupled_on}, fullwave, s_empty
    ),
}

gap_off = gap_map(coupled_off, fullwave, mask, s_empty)
gap_on = gap_map(coupled_on, fullwave, mask, s_empty)
np.savez_compressed(
    OUTPUT_DIR / "three_cube_320_gap_maps.npz",
    x=coupled_off.x,
    y=coupled_off.y,
    gap_off_db=gap_off,
    gap_on_db=gap_on,
    valid_mask=mask,
)
for label, gap in (("coupled_off", gap_off), ("coupled_on", gap_on)):
    finite = gap[np.isfinite(gap)]
    report[label]["gap_db"] = {
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "mean_abs": float(np.mean(np.abs(finite))),
    }

(OUTPUT_DIR / "three_cube_320_comparison.json").write_text(
    json.dumps(report, indent=2)
)

print(f"case {spec.case_id}  s_empty={s_empty:.4f}")
print(
    "empty-scene coherence |gamma| = "
    f"{empty_coherence['coherence_after_global_phase']:.4f}"
)
header = f"{'metric':34s} {'coupled OFF':>14s} {'coupled ON':>14s}"
print(header)
rows = (
    ("envelope NMSE", "envelope_nmse", ".4f"),
    ("magnitude correlation", "magnitude_correlation", ".4f"),
    ("magnitude RMSE dB", "magnitude_rmse_db", ".3f"),
    ("coherence after global phase", "coherence_after_global_phase", ".4f"),
)
for label, key, fmt in rows:
    off_value = report["coupled_off"][key]
    on_value = report["coupled_on"][key]
    print(f"{label:34s} {off_value:>14{fmt}} {on_value:>14{fmt}}")
for boundary in ("isb", "rsb"):
    off_value = report["coupled_off"][boundary]["p95_excess_jump_db"]
    on_value = report["coupled_on"][boundary]["p95_excess_jump_db"]
    print(f"{boundary.upper() + ' p95 excess dB':34s} {off_value:>14.3f} {on_value:>14.3f}")
for stat in ("median_db", "p99_db", "max_db"):
    off_value = report["coupled_off"]["det_only_jumps"][stat]
    on_value = report["coupled_on"]["det_only_jumps"][stat]
    print(f"{'det-only jump ' + stat:34s} {off_value:>14.3f} {on_value:>14.3f}")
for label in ("coupled_off", "coupled_on"):
    gap = report[label]["gap_db"]
    print(
        f"{label} gap dB: median={gap['median']:+.2f} "
        f"p10={gap['p10']:+.2f} p90={gap['p90']:+.2f} mean|.|={gap['mean_abs']:.2f}"
    )