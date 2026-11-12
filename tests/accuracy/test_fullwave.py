# Copyright Xingyu Chen.
# Deterministic solver against stored Tidy3D full-wave maps, per case and field region.

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch

from benchmarks.fullwave_validation import reference_set as rs
from benchmarks.fullwave_validation.metrics import analyze_boundaries
from benchmarks.fullwave_validation.models import FieldMap

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Channel solves need CUDA")

# Interaction depth per case (default 2): three for the scenes whose fields need third-order chains.
_DEPTH = {"three_cube_z": 3, "trihedral_z": 3, "trihedral_x": 3, "double_edge_y": 3,
          "double_edge_z": 3, "wall_column_z": 3, "wall_column_y": 3, "canyon_z_3g5": 3,
          "canyon_x_3g5": 3}
_CASES = rs.case_ids("fit") + rs.case_ids("holdout") + rs.case_ids("coverage")

# Ceilings per (case, plane, region): (RMS dB magnitude error, complex NMSE), and per (case, plane,
# ISB/RSB) the p95 excess of adjacent-sample dB jumps. Each is the error measured with the current
# corner model plus headroom: RMS max(1.1x, +0.25 dB), NMSE max(1.2x, +0.01), jump excess +1 dB, rounded
# up. The reference is good to about 0.05 dB and 4 degrees (free-space calibration against the exact
# dipole field), so the headroom only absorbs harmless solver changes; a region that moves by more is
# a real accuracy change and needs a new measurement, never a wider ceiling. A ``None`` ceiling marks a
# metric that KNOWN_GAPS holds instead. The coverage cases (walls, multilayer wall, concrete cubes,
# trihedral corner, two screens, wall and column, street canyon; 2.4 and 3.5 GHz) follow the same rule.
TOLERANCES: dict[tuple[str, int, str], tuple[float | None, float | None]] = {
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
    ("cube_z", 0, "shadow"): (2.40, 0.105),
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
    ("three_cube_z", 0, "lit"): (1.15, 0.025),
    ("three_cube_z", 0, "shadow"): (4.05, 0.175),
    ("three_cube_z", 0, "transition"): (3.05, 0.045),
    ("three_cube_z", 0, "corner"): (2.05, 0.045),
    ("cube_dielectric_z", 0, "lit"): (1.05, 0.020),
    ("cube_dielectric_z", 0, "shadow"): (6.35, 1.385),
    ("cube_dielectric_z", 0, "transition"): (4.05, 0.115),
    ("cube_dielectric_z", 0, "corner"): (4.55, 0.095),
    ("cube_dielectric_z", 1, "lit"): (1.10, 0.020),
    ("cube_dielectric_z", 1, "transition"): (1.45, 0.040),
    ("cube_dielectric_z", 1, "corner"): (0.95, 0.020),
    ("wall_concrete_z", 0, "lit"): (0.95, 0.025),
    ("wall_concrete_z", 0, "shadow"): (1.10, 0.060),
    ("wall_concrete_z", 1, "lit"): (1.35, 0.020),
    ("wall_concrete_z", 1, "shadow"): (0.95, 0.065),
    ("wall_concrete_y", 0, "lit"): (1.40, 0.020),
    ("wall_concrete_y", 0, "shadow"): (1.00, 0.055),
    ("wall_concrete_y", 1, "lit"): (0.65, 0.015),
    ("wall_concrete_y", 1, "shadow"): (1.25, 0.070),
    ("wall_layered_z", 0, "lit"): (1.10, 0.025),
    ("wall_layered_z", 0, "shadow"): (1.95, 0.295),
    ("wall_layered_z", 1, "lit"): (1.45, 0.020),
    ("wall_layered_z", 1, "shadow"): (1.70, 0.420),
    ("wall_layered_y", 0, "lit"): (1.55, 0.020),
    ("wall_layered_y", 0, "shadow"): (2.25, 0.175),
    ("wall_layered_y", 1, "lit"): (1.05, 0.015),
    ("wall_layered_y", 1, "shadow"): (3.05, 0.300),
    ("wall_concrete_z_2g4", 0, "lit"): (1.30, 0.065),
    ("wall_concrete_z_2g4", 0, "shadow"): (1.15, 0.045),
    ("wall_concrete_z_2g4", 1, "lit"): (1.35, 0.030),
    ("wall_concrete_z_2g4", 1, "shadow"): (1.00, 0.040),
    ("cube_concrete_z", 0, "lit"): (0.65, 0.020),
    ("cube_concrete_z", 0, "shadow"): (4.35, 0.200),
    ("cube_concrete_z", 0, "transition"): (1.15, 0.030),
    ("cube_concrete_z", 0, "corner"): (2.00, 0.025),
    ("cube_concrete_z", 1, "lit"): (1.10, 0.020),
    ("cube_concrete_z", 1, "transition"): (1.15, 0.025),
    ("cube_concrete_z", 1, "corner"): (0.85, 0.020),
    ("trihedral_z", 0, "lit"): (0.95, 0.015),
    ("trihedral_z", 0, "shadow"): (0.60, 0.130),
    ("trihedral_z", 0, "transition"): (0.70, 0.015),
    ("trihedral_z", 0, "corner"): (1.05, 0.020),
    ("trihedral_z", 1, "lit"): (1.00, 0.020),
    ("trihedral_z", 1, "shadow"): (1.35, 0.040),
    ("trihedral_z", 1, "transition"): (0.85, 0.015),
    ("trihedral_z", 1, "corner"): (1.05, 0.035),
    ("trihedral_x", 0, "lit"): (1.00, 0.020),
    ("trihedral_x", 0, "transition"): (1.65, 0.045),
    ("trihedral_x", 0, "corner"): (1.05, 0.025),
    ("trihedral_x", 1, "lit"): (1.20, 0.020),
    ("trihedral_x", 1, "transition"): (1.75, 0.040),
    ("double_edge_y", 0, "lit"): (1.00, 0.015),
    ("double_edge_y", 0, "shadow"): (1.90, 0.165),
    ("double_edge_y", 1, "lit"): (0.55, 0.015),
    ("double_edge_y", 1, "shadow"): (0.45, 0.025),
    ("double_edge_z", 0, "lit"): (1.85, 0.025),
    ("double_edge_z", 0, "shadow"): (2.20, 0.095),
    ("double_edge_z", 1, "lit"): (0.40, 0.015),
    ("double_edge_z", 1, "shadow"): (0.35, 0.015),
    ("double_edge_z", 1, "transition"): (1.75, 0.040),
    ("wall_column_z", 0, "lit"): (1.35, 0.030),
    ("wall_column_z", 0, "shadow"): (2.60, 0.110),
    ("wall_column_z", 0, "transition"): (0.90, 0.015),
    ("wall_column_y", 0, "lit"): (1.40, 0.020),
    ("wall_column_y", 0, "shadow"): (1.85, 0.080),
    ("wall_column_y", 0, "transition"): (1.65, 0.030),
    ("canyon_z_3g5", 0, "lit"): (2.05, 0.050),
    ("canyon_z_3g5", 0, "transition"): (2.40, 0.025),
    ("canyon_z_3g5", 0, "corner"): (2.50, 0.025),
    ("canyon_z_3g5", 1, "lit"): (1.80, 0.075),
    ("canyon_z_3g5", 1, "shadow"): (2.70, 0.585),
    ("canyon_z_3g5", 1, "transition"): (2.25, 0.020),
    ("canyon_z_3g5", 1, "corner"): (2.50, 0.020),
    ("cube_concrete_x", 0, "lit"): (0.65, 0.015),
    ("cube_concrete_x", 0, "shadow"): (3.80, 0.205),
    ("cube_concrete_x", 0, "transition"): (1.15, 0.035),
    ("cube_concrete_x", 0, "corner"): (1.50, 0.020),
    ("cube_concrete_x", 1, "lit"): (0.80, 0.015),
    ("cube_concrete_x", 1, "transition"): (0.85, 0.020),
    ("cube_concrete_x", 1, "corner"): (1.10, 0.025),
    ("canyon_x_3g5", 0, "lit"): (2.90, 0.040),
    ("canyon_x_3g5", 0, "transition"): (3.25, 0.045),
    ("canyon_x_3g5", 0, "corner"): (3.55, 0.035),
    ("canyon_x_3g5", 1, "lit"): (3.40, 0.060),
    ("canyon_x_3g5", 1, "shadow"): (2.00, 0.515),
    ("canyon_x_3g5", 1, "transition"): (2.90, 0.060),
    ("canyon_x_3g5", 1, "corner"): (2.50, 0.050),
    ("double_edge_y", 1, "transition"): (1.30, 0.020),
    ("trihedral_x", 0, "shadow"): (1.95, 0.140),
    ("trihedral_x", 1, "shadow"): (2.15, 0.170),
    ("trihedral_x", 1, "corner"): (2.40, 0.600),
    ("canyon_z_3g5", 0, "shadow"): (2.95, 0.840),
    ("canyon_x_3g5", 0, "shadow"): (1.65, 1.030),
}
JUMP_TOLERANCES: dict[tuple[str, int, str], float] = {
    ("canyon_z_3g5", 1, "ISB"): 2.10,
    ("wall_column_z", 0, "RSB"): 1.75,
    ("wall_column_y", 0, "RSB"): 2.00,
    ("double_edge_z", 1, "RSB"): 1.00,
    ("cube_concrete_z", 1, "RSB"): 1.50,
    ("cube_concrete_x", 1, "RSB"): 1.05,
    ("beam_z", 0, "ISB"): 0.90,
    ("beam_z", 0, "RSB"): 0.45,
    ("beam_z", 1, "ISB"): 1.50,
    ("beam_z", 1, "RSB"): 0.80,
    ("beam_y", 0, "ISB"): 1.10,
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
    ("cube_z", 0, "RSB"): 1.25,
    ("cube_z", 1, "RSB"): 1.80,
    ("cube_x", 0, "ISB"): 1.65,
    ("cube_x", 0, "RSB"): 0.75,
    ("cube_x", 1, "RSB"): 1.05,
    ("three_cube_z", 0, "ISB"): 2.90,
    ("three_cube_z", 0, "RSB"): 2.30,
    ("cube_dielectric_z", 0, "ISB"): 0.65,
    ("cube_dielectric_z", 0, "RSB"): 0.90,
    ("cube_dielectric_z", 1, "RSB"): 1.20,
    ("cube_concrete_z", 0, "ISB"): 1.10,
    ("cube_concrete_z", 0, "RSB"): 1.35,
    ("trihedral_z", 0, "ISB"): 1.25,
    ("trihedral_z", 0, "RSB"): 1.25,
    ("trihedral_z", 1, "ISB"): 0.85,
    ("trihedral_z", 1, "RSB"): 0.85,
    ("trihedral_x", 0, "ISB"): 0.85,
    ("trihedral_x", 0, "RSB"): 0.85,
    ("trihedral_x", 1, "ISB"): 0.25,
    ("trihedral_x", 1, "RSB"): 0.25,
    ("double_edge_y", 1, "ISB"): 1.00,
    ("double_edge_z", 1, "ISB"): 1.00,
    ("wall_column_z", 0, "ISB"): -0.55,
    ("wall_column_y", 0, "ISB"): -0.40,
    ("cube_concrete_x", 0, "ISB"): 1.25,
    ("cube_concrete_x", 0, "RSB"): 1.35,
    ("canyon_x_3g5", 1, "ISB"): 1.95,
}

# Errors the deterministic solver's model class does not explain: (case, plane, region or ISB/RSB,
# metric) -> (target, measured, diagnosis). Each runs as a strict xfail against its target, the error
# the model class reaches once the named defect is removed (measured with that defect worked around,
# or the matching single-object ceiling), so a fix shows up as an unexpected pass.
_CANYON_RSB = ("not isolated: the worst edges are depth-2 and depth-3 reflection paths that another "
               "block's corner blocks; their reflection-diffraction chain is enumerated on both sides "
               "but steps by less than the reflection (second-order chain boundaries)")
_BASE_CHAINS = ("the excess sits at the 59 RSB edges that are transmission-support but not LoS "
                "edges (p95 excess 14.6 / 13.6 dB): steps of the field transmitted through one block "
                "at another block's edge, which only transmission-then-diffraction chains compensate "
                "(an interaction class the solver does not have); the 471 RSB edges off transmission "
                "support show none")
_CANYON_ISB = ("the LoS steps are compensated to 1.4-2.3 % of the LoS (Channel-only probe, block "
               "interiors excluded); the excess comes from samples where the total is 15-20 dB below "
               "the LoS, which the dB jump magnifies")
KNOWN_GAPS: dict[tuple[str, int, str, str], tuple[float, float, str]] = {
    ("canyon_x_3g5", 0, "RSB", "jump"): (1.50, 2.98, _BASE_CHAINS),
    ("double_edge_y", 1, "RSB", "jump"): (1.50, 2.82, "the RSB is continuous; an interference null "
                                          "next to the transmitter lies 2-3 mm from its full-wave "
                                          "position on the four worst edges"),
    ("canyon_z_3g5", 0, "RSB", "jump"): (1.50, 3.98, _CANYON_RSB),
    ("canyon_z_3g5", 1, "RSB", "jump"): (1.50, 4.95, _CANYON_RSB),
    ("canyon_x_3g5", 1, "RSB", "jump"): (1.50, 2.32, _CANYON_RSB + "; plus " + _BASE_CHAINS),
    ("canyon_z_3g5", 0, "ISB", "jump"): (1.50, 2.25, _CANYON_ISB),
    ("canyon_x_3g5", 0, "ISB", "jump"): (1.50, 3.24, _CANYON_ISB),
}


def deterministic_plane(case: rs.ReferenceCase, plane: dict) -> tuple[np.ndarray, dict, float]:
    """Calibrated deterministic field and component maps on the reference samples, LoS spread."""

    from witwin.channel.radiomap import Coherent, solve

    result = solve(rs.build_channel_scene(case, plane), Coherent(
        components=frozenset({"los", "reflection", "diffraction", "transmission"}),
        max_depth=_DEPTH.get(case.case_id, 2), combine="field",
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


@functools.lru_cache(maxsize=None)
def _measured(case_id: str) -> tuple[dict, ...]:
    """Per plane: LoS calibration spread, region errors and boundary jump excess of one solve."""

    case = rs.load_reference_case(case_id)
    planes, _ = rs.load_reference(case)
    rows = []
    for plane in planes:
        field, components, los_spread = deterministic_plane(case, plane)
        rows.append({"los_spread": los_spread,
                     "regions": rs.region_errors(field, plane["field"], plane["regions"]),
                     "jumps": boundary_excess(case, plane, field, components)})
    return tuple(rows)


@pytest.mark.parametrize("case_id", _CASES)
def test_deterministic_matches_fullwave_reference(accuracy, case_id):
    for index, row in enumerate(_measured(case_id)):
        # The LoS term alone is the exact far field of the dipole up to one complex constant.
        accuracy.check(f"{case_id}/plane{index}/los-calibration", row["los_spread"], 1e-3)
        for region, errors in row["regions"].items():
            ceiling = TOLERANCES.get((case_id, index, region))
            assert ceiling is not None, f"no tolerance for {case_id}/plane{index}/{region}: {errors}"
            for metric, limit in zip(("rmse_db", "nmse"), ceiling):
                if (case_id, index, region, metric) in KNOWN_GAPS:
                    continue
                assert limit is not None, f"no {metric} ceiling for {case_id}/plane{index}/{region}"
                accuracy.check(f"{case_id}/plane{index}/{region}/{metric}", errors[metric], limit)
        for kind, excess in row["jumps"].items():
            if (case_id, index, kind, "jump") in KNOWN_GAPS:
                continue
            ceiling = JUMP_TOLERANCES.get((case_id, index, kind))
            assert ceiling is not None, f"no jump tolerance for {case_id}/plane{index}/{kind}: {excess}"
            accuracy.check(f"{case_id}/plane{index}/{kind}/p95_excess_jump_db", excess, ceiling)


@pytest.mark.parametrize("gap", [
    pytest.param(key, id="/".join(map(str, key)), marks=pytest.mark.xfail(
        strict=True, reason=f"measured {measured}; {diagnosis}"))
    for key, (_, measured, diagnosis) in KNOWN_GAPS.items()])
def test_deterministic_fullwave_known_gaps(accuracy, gap):
    case_id, index, name, metric = gap
    row = _measured(case_id)[index]
    value = row["jumps"][name] if metric == "jump" else row["regions"][name][metric]
    accuracy.check("/".join(map(str, gap)), value, KNOWN_GAPS[gap][0])
