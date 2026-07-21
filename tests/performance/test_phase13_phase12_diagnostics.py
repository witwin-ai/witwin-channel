from __future__ import annotations

import numpy as np
import pytest

from benchmarks.phase13_phase12.contracts import EvidenceError
from benchmarks.phase13_phase12.diagnostics import (
    _diffraction_oracle,
    _semantic_hash,
    load_diagnostic_contract,
)


def _source_lane_fixture() -> dict[str, np.ndarray]:
    valid = np.array([[True, False, True], [False, True, True]], dtype=np.bool_)
    rows: dict[str, np.ndarray] = {
        "valid": valid.reshape(-1),
        "failure": np.zeros((1,), dtype=np.int32),
        "num_paths": np.array([4], dtype=np.int32),
    }
    for index, name in enumerate(("x_re", "x_im", "y_re", "y_im", "z_re", "z_im")):
        rows[name] = (
            np.arange(6, dtype=np.float32).reshape(2, 3) + np.float32(index)
        ).reshape(-1)
    target = np.empty((2, 6), dtype=np.float32)
    for index, name in enumerate(("x_re", "x_im", "y_re", "y_im", "z_re", "z_im")):
        values = rows[name].reshape(2, 3)
        target[:, index] = np.where(valid, values, 0.0).sum(axis=1, dtype=np.float32)
    rows["target"] = target
    return rows


def test_diagnostic_contract_freezes_stable_owner_and_source_lane_availability() -> None:
    contract = load_diagnostic_contract()
    groups = contract["groups"]
    assert groups["diffraction"]["variants"]["baseline"]["source_lane_available"] is False
    assert groups["diffraction"]["variants"]["candidate"]["source_lane_available"] is True


def test_float64_oracle_uses_pair_major_state_fast_valid_rows() -> None:
    result = _diffraction_oracle(
        _source_lane_fixture(), {"pair_count": 2, "state_capacity": 3}
    )
    assert result["valid_count"] == 4
    assert result["target_vs_float64"] == {
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "max_ulp_error": 0,
    }


def test_float64_oracle_rejects_partial_result_failure() -> None:
    arrays = _source_lane_fixture()
    arrays["failure"][0] = 1
    with pytest.raises(EvidenceError, match="capacity failure"):
        _diffraction_oracle(arrays, {"pair_count": 2, "state_capacity": 3})


def test_semantic_hash_binds_dtype_and_shape() -> None:
    array = np.arange(4, dtype=np.float32)
    assert _semantic_hash({"value": array}) != _semantic_hash(
        {"value": array.reshape(2, 2)}
    )
    assert _semantic_hash({"value": array}) != _semantic_hash(
        {"value": array.astype(np.float64)}
    )
