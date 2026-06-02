"""Regression tests for validation-path diffraction state audits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt
import pytest

from witwin.channel.validation import (
    build_double_wedge_case,
    run_diffraction_order_sweep,
    sweep_to_npz_payload,
)


pytestmark = [pytest.mark.gpu, pytest.mark.acceptance, pytest.mark.validation]


def test_validation_sweep_exports_state_audit():
    case = build_double_wedge_case()
    sweep = run_diffraction_order_sweep(case=case, frequency=1e9, grid_size=24, max_order=2)

    order2 = sweep["orders"][1]
    audit = order2["state_audit"]
    assert audit is not None
    assert audit["n_states"] > 0
    assert audit["history_size"] == 2

    state_order = audit["order"]
    assert np.any(state_order == 1)
    assert np.any(state_order == 2)

    first_order_mask = state_order == 1
    second_order_mask = state_order == 2

    assert np.all(audit["path_edge_idx_0"][first_order_mask] >= 0)
    assert np.all(audit["path_edge_idx_1"][first_order_mask] == -1)
    assert np.all(audit["path_edge_idx_0"][second_order_mask] >= 0)
    assert np.all(audit["path_edge_idx_1"][second_order_mask] >= 0)
    assert np.all(audit["edge_idx"][second_order_mask] == audit["path_edge_idx_1"][second_order_mask])
    assert np.all(audit["source_type"][first_order_mask] == "direct_tx")
    assert np.all(audit["source_type"][second_order_mask] == "direct_tx")
    assert np.all(audit["prefix_reflection_depth"] == 0)
    assert np.all(audit["suffix_reflection_depth"] == 0)
    assert np.all(audit["path_sequence"][first_order_mask] == "S -> D")
    assert np.all(audit["path_sequence"][second_order_mask] == "S -> D -> D")
    assert np.all(audit["approximation_mode"][first_order_mask] == "exact_direct_first_order")
    assert np.all(audit["approximation_mode"][second_order_mask] == "approx_recursive_diffraction")


def test_validation_npz_payload_contains_audit_arrays():
    case = build_double_wedge_case()
    sweep = run_diffraction_order_sweep(case=case, frequency=1e9, grid_size=16, max_order=2)
    payload = sweep_to_npz_payload(sweep)

    assert "order_2_audit_order" in payload
    assert "order_2_audit_path_edge_idx_0" in payload
    assert "order_2_audit_path_edge_idx_1" in payload
    assert "order_2_audit_path_sequence" in payload
    assert "order_2_audit_approximation_mode" in payload
    assert int(payload["order_2_audit_history_size"]) == 2


