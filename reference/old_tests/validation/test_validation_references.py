"""Regression tests for explicit finite-wedge diffraction references."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import witwin as wt
import pytest

from witwin.channel.validation import (
    build_double_wedge_case,
    build_single_wedge_case,
    build_triple_wedge_case,
    compare_first_order_overlap_against_sionna,
    evaluate_closed_form_double_diffraction_reference,
    evaluate_closed_form_triple_diffraction_reference,
)


pytestmark = [pytest.mark.gpu, pytest.mark.acceptance, pytest.mark.validation]


def test_double_diffraction_reference_matches_explicit_pair_expansion():
    result = evaluate_closed_form_double_diffraction_reference(
        build_double_wedge_case(),
        frequency=1e9,
        grid_size=24,
    )

    assert result["max_abs_complex_error"] < 2e-10
    assert result["rms_complex_error"] < 1e-11

    reference_audit = result["reference_state_audit"]
    assert reference_audit["n_states"] > 0
    assert np.all(reference_audit["order"] == 2)
    assert np.all(reference_audit["path_sequence"] == "S -> D -> D")

    cut_error = np.max(
        np.abs(result["candidate_line_cut"]["values"] - result["reference_line_cut"]["values"])
    )
    assert float(cut_error) < 2e-10


def test_first_order_overlap_matches_explicit_finite_wedge_reference():
    result = compare_first_order_overlap_against_sionna(
        build_single_wedge_case(),
        frequency=1e9,
        grid_size=24,
    )

    assert result["max_abs_complex_error"] < 2e-9
    assert result["rms_complex_error"] < 5e-10

    candidate_audit = result["candidate_state_audit"]
    assert candidate_audit["n_states"] > 0
    assert np.all(candidate_audit["order"] == 1)
    assert np.all(candidate_audit["path_sequence"] == "S -> D")


def test_triple_diffraction_reference_matches_explicit_triplet_expansion():
    result = evaluate_closed_form_triple_diffraction_reference(
        build_triple_wedge_case(),
        frequency=1e9,
        grid_size=20,
    )

    assert result["max_abs_complex_error"] < 5e-10
    assert result["rms_complex_error"] < 1e-10

    reference_audit = result["reference_state_audit"]
    assert reference_audit["n_states"] > 0
    assert np.all(reference_audit["order"] == 3)
    assert np.all(reference_audit["path_sequence"] == "S -> D -> D -> D")

    cut_error = np.max(
        np.abs(result["candidate_line_cut"]["values"] - result["reference_line_cut"]["values"])
    )
    assert float(cut_error) < 1e-10


