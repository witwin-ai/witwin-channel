import pytest
import torch

from witwin.channel_native.core.runtime.assignments import AssignmentStore
from witwin.channel_native.core.runtime.geometry import GeometryStore
from witwin.channel_native.core.runtime.material_store import MaterialStore


def test_geometry_store_rejects_wrong_vertex_shape():
    with pytest.raises(ValueError, match="vertices"):
        GeometryStore(
            vertices=torch.zeros((3,), dtype=torch.float32),
            faces=torch.zeros((1, 3), dtype=torch.int32),
            face_normals=torch.zeros((1, 3), dtype=torch.float32),
            edges=torch.zeros((3, 2), dtype=torch.int32),
            edge_adj_faces=torch.zeros((3, 2), dtype=torch.int32),
            edge_param_range=torch.zeros((3, 2), dtype=torch.float32),
            face_structure_id=torch.zeros((1,), dtype=torch.int32),
            face_surface_id=torch.zeros((1,), dtype=torch.int32),
            version=0,
        )


def test_material_store_rejects_per_face_parameter_expansion():
    with pytest.raises(ValueError, match="same length"):
        MaterialStore(
            eps_r=torch.ones((2,), dtype=torch.float32),
            mu_r=torch.ones((1,), dtype=torch.float32),
            sigma_e=torch.zeros((1,), dtype=torch.float32),
            gain=torch.ones((1,), dtype=torch.float32),
            model_id=torch.ones((1,), dtype=torch.int32),
            model_params=torch.zeros((1, 4), dtype=torch.float32),
            frequency_hz=3.5e9,
            version=0,
        )


def test_assignment_store_validates_face_material_length():
    with pytest.raises(ValueError, match="face_material_id"):
        AssignmentStore(
            face_material_id=torch.zeros((2,), dtype=torch.int32),
            edge_material_id0=torch.zeros((3,), dtype=torch.int32),
            edge_material_id1=torch.zeros((3,), dtype=torch.int32),
            surface_material_id=torch.zeros((1,), dtype=torch.int32),
            structure_material_id=torch.zeros((1,), dtype=torch.int32),
            num_faces=1,
            num_edges=3,
            version=0,
        )
