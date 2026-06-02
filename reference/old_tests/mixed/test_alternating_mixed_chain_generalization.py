"""Regression tests for generalized alternating mixed diffraction chains."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import witwin as wt

from witwin.channel import FieldMonitor, Tracer
from witwin.channel.validation import build_double_wedge_case
def test_inserted_reflection_budget_controls_alternating_chain_depth():
    case = build_double_wedge_case()
    common_kwargs = dict(
        frequency=1e9,
        scene=case.scene,
        reflection_n_rays=4096,
        reflection_max_bounces=1,
        enable_rd_diffraction=True,
        max_diffractions=3,
    )

    bounded = Tracer(
        max_inserted_reflections_per_path=1,
        **common_kwargs,
    )
    generalized = Tracer(
        max_inserted_reflections_per_path=2,
        **common_kwargs,
    )
    monitor = FieldMonitor(
        "validation_plane",
        axis="z",
        position=case.calculation_height,
        bounds=(case.range_x, case.range_y),
        grid_size=20,
    )

    bounded_result = bounded.trace(
        wt.Point3f(*case.tx_pos),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )
    generalized_result = generalized.trace(
        wt.Point3f(*case.tx_pos),
        monitor=monitor,
        verbose=False,
        return_diffraction_audit=True,
    )

    bounded_sequences = set(bounded_result.primary.diffraction_detail["state_audit"]["path_sequence"])
    generalized_audit = generalized_result.primary.diffraction_detail["state_audit"]
    generalized_sequences = set(generalized_audit["path_sequence"])

    target_sequence = "S -> D -> R -> D -> R -> D"
    assert target_sequence not in bounded_sequences
    assert target_sequence in generalized_sequences

    target_idx = next(
        idx
        for idx, label in enumerate(generalized_audit["path_sequence"])
        if label == target_sequence
    )
    reflection_history = [
        int(generalized_audit[f"path_reflection_depth_{slot}"][target_idx])
        for slot in range(generalized_audit["history_size"])
    ]
    assert reflection_history == [0, 1, 1]

    solver_metadata = generalized_result.primary.metadata
    assert solver_metadata["path_families"]["Arbitrary alternating mixed chains"]["status"] == "approximate"
    assert solver_metadata["mixed_chain_budget"]["max_inserted_reflections_per_path"] == 2


