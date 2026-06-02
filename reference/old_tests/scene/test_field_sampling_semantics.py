from __future__ import annotations

import numpy as np
import pytest

import witwin as wt
from witwin.channel import Field
@pytest.mark.gpu
def test_field_coordinates_match_legacy_boundary_points():
    field = Field(bounds=((0.0, 4.0), (10.0, 16.0)), size=(4, 3))
    coords = field.get_coordinates()

    x_coords = np.asarray(coords["x_coords"], dtype=np.float32)
    y_coords = np.asarray(coords["y_coords"], dtype=np.float32)

    np.testing.assert_allclose(x_coords, np.array([0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(y_coords, np.array([10.0, 13.0, 16.0], dtype=np.float32))


@pytest.mark.gpu
def test_field_coordinates_are_not_cell_centers():
    field = Field(bounds=((0.0, 4.0), (0.0, 4.0)), size=(4, 4))
    coords = field.get_coordinates()

    x_coords = np.asarray(coords["x_coords"], dtype=np.float32)
    center_coords = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    assert not np.allclose(x_coords, center_coords)
    np.testing.assert_allclose(x_coords, np.array([0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0], dtype=np.float32))


@pytest.mark.gpu
def test_field_pos_to_idx_preserves_legacy_span_over_n_binning():
    field = Field(bounds=((0.0, 4.0), (0.0, 4.0)), size=(4, 4))

    x = wt.Float([0.1, 1.1, 2.1, 3.9])
    y = wt.Float([0.1, 1.1, 2.1, 3.9])
    idx = np.asarray(field.pos_to_idx(x, y), dtype=np.uint32)

    np.testing.assert_array_equal(idx, np.array([0, 5, 10, 15], dtype=np.uint32))
