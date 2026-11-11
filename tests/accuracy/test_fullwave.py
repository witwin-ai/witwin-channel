# Copyright Xingyu Chen.
# Deterministic solver against stored Tidy3D full-wave maps, per case and field region.

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.fullwave_validation import reference_set as rs
from benchmarks.fullwave_validation.metrics import analyze_boundaries
from benchmarks.fullwave_validation.models import FieldMap

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")

_DEPTH = {"three_cube_z": 3}
_CASES = rs.case_ids("fit") + rs.case_ids("holdout")

# Ceilings per (case, plane, region): (RMS dB magnitude error, complex NMSE), and per (case, plane,
# ISB/RSB) the p95 excess of adjacent-sample dB jumps. Each is the error measured with the current
# corner model plus headroom: RMS max(1.1x, +0.25 dB), NMSE max(1.2x, +0.01), jump excess +1 dB, rounded
# up. The reference is good to about 0.05 dB and 4 degrees (free-space calibration against the exact
# dipole field), so the headroom only absorbs harmless solver changes; a region that moves by more is
# a real accuracy change and needs a new measurement, never a wider ceiling.
TOLERANCES: dict[tuple[str, int, str], tuple[float, float]] = {
    ("beam_z", 0, "lit"): (0.85, 0.020),
    ("beam_z", 0, "shadow"): (1.80, 0.060),
    ("beam_z", 0, "transition"): (1.60, 0.035),
    ("beam_z", 0, "corner"): (2.35, 0.030),
    ("beam_z", 1, "lit"): (1.60, 0.025),
    ("beam_z", 1, "shadow"): (1.35, 0.035),
    ("beam_z", 1, "transition"): (0.95, 0.025),
    ("beam_z", 1, "corner"): (1.20, 0.025),
    ("beam_y", 0, "lit"): (0.65, 0.015),
    ("beam_y", 0, "shadow"): (3.30, 0.165),
    ("beam_y", 0, "transition"): (1.00, 0.020),
    ("beam_y", 0, "corner"): (1.55, 0.025),
    ("beam_y", 1, "lit"): (1.00, 0.015),
    ("beam_y", 1, "shadow"): (2.35, 0.185),
    ("beam_y", 1, "transition"): (1.20, 0.020),
    ("beam_y", 1, "corner"): (1.80, 0.025),
    ("halfplane_z", 0, "lit"): (0.90, 0.015),
    ("halfplane_z", 0, "shadow"): (0.75, 0.020),
    ("halfplane_z", 0, "transition"): (0.80, 0.015),
    ("halfplane_z", 0, "corner"): (1.30, 0.020),
    ("halfplane_z", 1, "lit"): (0.65, 0.015),
    ("halfplane_z", 1, "shadow"): (1.00, 0.020),
    ("halfplane_z", 1, "transition"): (0.90, 0.015),
    ("halfplane_z", 1, "corner"): (0.90, 0.015),
    ("halfplane_y", 0, "lit"): (0.55, 0.015),
    ("halfplane_y", 0, "shadow"): (1.05, 0.020),
    ("halfplane_y", 0, "transition"): (0.70, 0.015),
    ("halfplane_y", 0, "corner"): (0.85, 0.015),
    ("halfplane_y", 1, "lit"): (0.50, 0.015),
    ("halfplane_y", 1, "shadow"): (1.15, 0.020),
    ("halfplane_y", 1, "transition"): (0.70, 0.015),
    ("halfplane_y", 1, "corner"): (0.60, 0.015),
    ("cube_z", 0, "lit"): (0.80, 0.020),
    ("cube_z", 0, "shadow"): (3.25, 0.140),
    ("cube_z", 0, "transition"): (1.45, 0.035),
    ("cube_z", 0, "corner"): (2.00, 0.030),
    ("cube_z", 1, "lit"): (1.05, 0.020),
    ("cube_z", 1, "transition"): (1.20, 0.030),
    ("cube_z", 1, "corner"): (0.85, 0.020),
    ("cube_x", 0, "lit"): (1.05, 0.020),
    ("cube_x", 0, "shadow"): (3.75, 0.105),
    ("cube_x", 0, "transition"): (1.35, 0.030),
    ("cube_x", 0, "corner"): (1.65, 0.025),
    ("cube_x", 1, "lit"): (1.00, 0.020),
    ("cube_x", 1, "transition"): (0.75, 0.020),
    ("cube_x", 1, "corner"): (1.35, 0.030),
    ("three_cube_z", 0, "lit"): (1.20, 0.025),
    ("three_cube_z", 0, "shadow"): (4.20, 0.175),
    ("three_cube_z", 0, "transition"): (3.05, 0.045),
    ("three_cube_z", 0, "corner"): (2.10, 0.045),
    ("cube_dielectric_z", 0, "lit"): (1.05, 0.020),
    ("cube_dielectric_z", 0, "shadow"): (6.35, 1.385),
    ("cube_dielectric_z", 0, "transition"): (4.05, 0.125),
    ("cube_dielectric_z", 0, "corner"): (4.55, 0.095),
    ("cube_dielectric_z", 1, "lit"): (1.10, 0.020),
    ("cube_dielectric_z", 1, "transition"): (1.55, 0.050),
    ("cube_dielectric_z", 1, "corner"): (1.00, 0.020),
}
JUMP_TOLERANCES: dict[tuple[str, int, str], float] = {
    ("beam_z", 0, "ISB"): 1.10,
    ("beam_z", 0, "RSB"): 0.45,
    ("beam_z", 1, "ISB"): 1.50,
    ("beam_z", 1, "RSB"): 0.80,
    ("beam_y", 0, "ISB"): 1.15,
    ("beam_y", 0, "RSB"): 0.95,
    ("beam_y", 1, "ISB"): 1.45,
    ("beam_y", 1, "RSB"): 1.55,
    ("halfplane_z", 0, "ISB"): 0.95,
    ("halfplane_z", 0, "RSB"): 0.55,
    ("halfplane_z", 1, "ISB"): 1.00,
    ("halfplane_z", 1, "RSB"): 1.05,
    ("halfplane_y", 0, "ISB"): 1.00,
    ("halfplane_y", 0, "RSB"): 1.00,
    ("halfplane_y", 1, "ISB"): 0.90,
    ("halfplane_y", 1, "RSB"): 1.25,
    ("cube_z", 0, "ISB"): 1.15,
    ("cube_z", 0, "RSB"): 1.45,
    ("cube_z", 1, "RSB"): 1.80,
    ("cube_x", 0, "ISB"): 1.70,
    ("cube_x", 0, "RSB"): 0.75,
    ("cube_x", 1, "RSB"): 1.05,
    ("three_cube_z", 0, "ISB"): 2.90,
    ("three_cube_z", 0, "RSB"): 2.70,
    ("cube_dielectric_z", 0, "ISB"): 5.10,
    ("cube_dielectric_z", 0, "RSB"): 2.05,
    ("cube_dielectric_z", 1, "RSB"): 4.15,
}


def deterministic_plane(case: rs.ReferenceCase, plane: dict) -> tuple[np.ndarray, dict, float]:
    """Calibrated deterministic field and component maps on the reference samples, LoS spread."""

    from witwin.channel.deterministic import Config, solve

    result = solve(rs.build_channel_scene(case, plane), Config(
        components=frozenset({"los", "reflection", "diffraction", "transmission"}),
        max_depth=_DEPTH.get(case.case_id, 2), coherent=True,
        return_field=True, coupled_paths=True, coupled_candidate_limit=1_000_000,
        isb_boundary_taper=False), reference_frequency_hz=case.frequency_hz)
    # The grid is shape (x.size, y.size), so solver maps are [1, x, y]; the planes are (y, x).
    shape = plane["field"].shape

    def plane_layout(values) -> np.ndarray:
        array = values.detach().cpu().numpy()
        if array.shape != (1, shape[1], shape[0]):
            raise AssertionError(f"unexpected solver map shape {array.shape} for plane {shape}")
        return array[0].T

    field = plane_layout(result.field)
    los = plane_layout(result.component_fields["los"])
    far = rs.dipole_field(case, rs.plane_points(plane["x"], plane["y"], plane["z"]), near_field=False)
    visible = np.abs(los) > 0
    ratio = los[visible] / far[visible]
    scale = complex(np.median(ratio.real), np.median(ratio.imag))
    components = {name: plane_layout(values) / scale
                  for name, values in result.component_fields.items() if values.numel() > 0}
    return field / scale, components, float(np.max(np.abs(ratio / scale - 1.0)))


def boundary_excess(
    case: rs.ReferenceCase, plane: dict, field: np.ndarray, components: dict,
) -> dict[str, float]:
    """p95 excess of adjacent-sample dB jumps across LoS (ISB) and reflection (RSB) support edges."""

    metadata = {"case_id": case.case_id, "case_fingerprint": case.fingerprint,
                "frequency_hz": case.frequency_hz}
    deterministic = FieldMap(plane["x"], plane["y"], field, components=components, metadata=metadata)
    reference = FieldMap(plane["x"], plane["y"], plane["field"], metadata=metadata)
    metrics = analyze_boundaries(deterministic, reference, valid_mask=plane["regions"] != rs.INVALID)
    return {kind: row.p95_excess_jump_db for kind, row in metrics.items() if row.edge_count}


@pytest.mark.parametrize("case_id", _CASES)
def test_deterministic_matches_fullwave_reference(accuracy, case_id):
    case = rs.load_reference_case(case_id)
    planes, _ = rs.load_reference(case)
    for index, plane in enumerate(planes):
        field, components, los_spread = deterministic_plane(case, plane)
        # The LoS term alone is the exact far field of the dipole up to one complex constant.
        accuracy.check(f"{case_id}/plane{index}/los-calibration", los_spread, 1e-3)
        errors = rs.region_errors(field, plane["field"], plane["regions"])
        for region, row in errors.items():
            ceiling = TOLERANCES.get((case_id, index, region))
            assert ceiling is not None, f"no tolerance for {case_id}/plane{index}/{region}: {row}"
            accuracy.check(f"{case_id}/plane{index}/{region}/rmse_db", row["rmse_db"], ceiling[0])
            accuracy.check(f"{case_id}/plane{index}/{region}/nmse", row["nmse"], ceiling[1])
        for kind, excess in boundary_excess(case, plane, field, components).items():
            ceiling = JUMP_TOLERANCES.get((case_id, index, kind))
            assert ceiling is not None, f"no jump tolerance for {case_id}/plane{index}/{kind}: {excess}"
            accuracy.check(f"{case_id}/plane{index}/{kind}/p95_excess_jump_db", excess, ceiling)
