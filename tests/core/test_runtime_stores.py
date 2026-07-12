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
            material_id=torch.arange(1, dtype=torch.int32),
            eps_r=torch.ones((2,), dtype=torch.float32),
            mu_r=torch.ones((1,), dtype=torch.float32),
            sigma_e=torch.zeros((1,), dtype=torch.float32),
            gain=torch.ones((1,), dtype=torch.float32),
            model_id=torch.ones((1,), dtype=torch.int32),
            thickness_m=torch.ones((1,), dtype=torch.float32),
            scattering_coefficient=torch.zeros((1,), dtype=torch.float32),
            xpd_coefficient=torch.zeros((1,), dtype=torch.float32),
            layer_offset=torch.zeros((1,), dtype=torch.int32),
            layer_count=torch.ones((1,), dtype=torch.int32),
            layer_thickness_m=torch.ones((1,), dtype=torch.float32),
            layer_eps_r=torch.ones((1,), dtype=torch.float32),
            layer_sigma_e=torch.zeros((1,), dtype=torch.float32),
            layer_mu_r=torch.ones((1,), dtype=torch.float32),
            rough_sigma_h_m=torch.zeros((1,), dtype=torch.float32),
            rough_corr_x_m=torch.zeros((1,), dtype=torch.float32),
            rough_corr_y_m=torch.zeros((1,), dtype=torch.float32),
            rough_axis_rad=torch.zeros((1,), dtype=torch.float32),
            geometry_mode_id=torch.zeros((1,), dtype=torch.int32),
            scatter_model_id=torch.zeros((1,), dtype=torch.int32),
            material_keys=("0:test:test",),
            frequency_hz=3.5e9,
            abi_version=3,
            cache_token="test",
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
