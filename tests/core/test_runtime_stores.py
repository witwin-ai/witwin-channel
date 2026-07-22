import inspect
from dataclasses import replace

import pytest
import torch

from witwin.channel.core.runtime import _validation as legacy_validation
from witwin.channel.core.runtime.assignments import AssignmentStore
from witwin.channel.core.runtime.geometry import GeometryStore
from witwin.channel.core.runtime.material_store import MaterialStore
from witwin.channel.scene.stores import _validation as canonical_validation
from witwin.channel.scene.stores.assignments import (
    AssignmentStore as CanonicalAssignmentStore,
)
from witwin.channel.scene.stores.geometry import (
    GeometryStore as CanonicalGeometryStore,
)
from witwin.channel.scene.stores.materials import (
    MaterialStore as CanonicalMaterialStore,
)


_REQUIRE_TENSOR_SOURCE = '''def require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    ndim: int,
    trailing_shape: tuple[int, ...] = (),
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if trailing_shape and tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
        raise ValueError(f"{name} must end with shape {trailing_shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
'''


def test_runtime_tensor_validation_owner_and_body_are_exact():
    assert legacy_validation.require_tensor is canonical_validation.require_tensor
    assert inspect.getsource(canonical_validation.require_tensor) == _REQUIRE_TENSOR_SOURCE


def test_runtime_tensor_validation_order_is_exact():
    require_tensor = canonical_validation.require_tensor
    with pytest.raises(TypeError, match="value must be a torch.Tensor"):
        require_tensor("value", object(), dtype=torch.float32, ndim=2)
    with pytest.raises(TypeError, match="value must have dtype torch.float32"):
        require_tensor(
            "value", torch.zeros((2,), dtype=torch.int32), dtype=torch.float32, ndim=2
        )
    with pytest.raises(ValueError, match="value must have 2 dimensions"):
        require_tensor("value", torch.zeros((2,)), dtype=torch.float32, ndim=2)
    with pytest.raises(ValueError, match="value must end with shape"):
        require_tensor(
            "value",
            torch.zeros((2, 3)),
            dtype=torch.float32,
            ndim=2,
            trailing_shape=(2,),
        )
    with pytest.raises(ValueError, match="value must be contiguous"):
        require_tensor(
            "value",
            torch.zeros((2, 3)).transpose(0, 1),
            dtype=torch.float32,
            ndim=2,
            trailing_shape=(2,),
        )


def test_geometry_store_canonical_owner_preserves_input_storage():
    tensors = {
        "vertices": torch.zeros((3, 3), dtype=torch.float32),
        "faces": torch.zeros((1, 3), dtype=torch.int32),
        "face_normals": torch.zeros((1, 3), dtype=torch.float32),
        "edges": torch.zeros((3, 2), dtype=torch.int32),
        "edge_adj_faces": torch.zeros((3, 2), dtype=torch.int32),
        "edge_param_range": torch.zeros((3, 2), dtype=torch.float32),
        "face_structure_id": torch.zeros((1,), dtype=torch.int32),
        "face_surface_id": torch.zeros((1,), dtype=torch.int32),
    }
    store = CanonicalGeometryStore(
        **tensors, structure_uv_presence=((True, True),), version=0
    )

    assert CanonicalGeometryStore is GeometryStore
    for name, tensor in tensors.items():
        stored = getattr(store, name)
        assert stored is tensor
        assert stored.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


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
            structure_uv_presence=((False, False),),
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


def _material_store_kwargs() -> dict[str, object]:
    return {
        "material_id": torch.arange(2, dtype=torch.int32),
        "eps_r": torch.tensor([2.0, 3.0], dtype=torch.float32),
        "mu_r": torch.ones((2,), dtype=torch.float32),
        "sigma_e": torch.zeros((2,), dtype=torch.float32),
        "gain": torch.ones((2,), dtype=torch.float32),
        "model_id": torch.ones((2,), dtype=torch.int32),
        "thickness_m": torch.ones((2,), dtype=torch.float32),
        "scattering_coefficient": torch.zeros((2,), dtype=torch.float32),
        "xpd_coefficient": torch.zeros((2,), dtype=torch.float32),
        "layer_offset": torch.tensor([0, 1], dtype=torch.int32),
        "layer_count": torch.tensor([1, 2], dtype=torch.int32),
        "layer_thickness_m": torch.ones((3,), dtype=torch.float32),
        "layer_eps_r": torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32),
        "layer_sigma_e": torch.zeros((3,), dtype=torch.float32),
        "layer_mu_r": torch.ones((3,), dtype=torch.float32),
        "rough_sigma_h_m": torch.zeros((2,), dtype=torch.float32),
        "rough_corr_x_m": torch.zeros((2,), dtype=torch.float32),
        "rough_corr_y_m": torch.zeros((2,), dtype=torch.float32),
        "rough_axis_rad": torch.zeros((2,), dtype=torch.float32),
        "geometry_mode_id": torch.zeros((2,), dtype=torch.int32),
        "scatter_model_id": torch.zeros((2,), dtype=torch.int32),
        "material_keys": ("0:test:a", "1:test:b"),
        "frequency_hz": 3.5e9,
        "abi_version": 3,
        "cache_token": "test",
        "version": 0,
    }


def _material_store() -> CanonicalMaterialStore:
    return CanonicalMaterialStore(**_material_store_kwargs())


def test_material_store_canonical_owner_preserves_input_storage_and_scalar_view():
    values = _material_store_kwargs()
    store = CanonicalMaterialStore(**values)

    assert CanonicalMaterialStore is MaterialStore
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            stored = getattr(store, name)
            assert stored is value
            assert stored.untyped_storage().data_ptr() == (
                value.untyped_storage().data_ptr()
            )
    assert store.eps_r.tolist() == store.layer_eps_r[store.layer_offset].tolist()
    assert store.eps_r.untyped_storage().data_ptr() != (
        store.layer_eps_r.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {
                "eps_r": torch.ones((3,), dtype=torch.float32),
                "layer_eps_r": torch.ones((4,), dtype=torch.float32),
            },
            "material tensors must have the same length",
        ),
        (
            {
                "layer_eps_r": torch.ones((4,), dtype=torch.float32),
                "layer_count": torch.tensor([0, 3], dtype=torch.int32),
            },
            "layer tensors must have the same length",
        ),
        (
            {
                "layer_count": torch.tensor([0, 3], dtype=torch.int32),
                "layer_offset": torch.tensor([1, 0], dtype=torch.int32),
            },
            "layer_count must be >= 1 for every material",
        ),
        (
            {
                "layer_offset": torch.tensor([0, 2], dtype=torch.int32),
                "frequency_hz": 0.0,
            },
            "layer_offset must be the exclusive scan of layer_count",
        ),
        (
            {
                "layer_count": torch.tensor([1, 1], dtype=torch.int32),
                "frequency_hz": 0.0,
            },
            "layer_count must sum to the layer tensor length",
        ),
        (
            {"frequency_hz": 0.0, "abi_version": 2},
            "frequency_hz must be positive",
        ),
        (
            {
                "abi_version": 2,
                "material_id": torch.tensor([1, 0], dtype=torch.int32),
            },
            "MaterialStore requires material ABI version 3",
        ),
        (
            {
                "material_id": torch.tensor([1, 0], dtype=torch.int32),
                "material_keys": ("same", "same"),
            },
            "material_id must be dense and stable",
        ),
        ({"material_keys": ("same", "same")}, "material_keys must be unique"),
    ),
)
def test_material_store_csr_and_abi_validation_order_is_exact(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(_material_store(), **changes)


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


def _assignment_store_kwargs() -> dict[str, object]:
    return {
        "face_material_id": torch.zeros((1,), dtype=torch.int32),
        "edge_material_id0": torch.zeros((3,), dtype=torch.int32),
        "edge_material_id1": torch.zeros((3,), dtype=torch.int32),
        "surface_material_id": torch.zeros((1,), dtype=torch.int32),
        "structure_material_id": torch.zeros((1,), dtype=torch.int32),
        "num_faces": 1,
        "num_edges": 3,
        "version": 0,
    }


def test_assignment_store_canonical_owner_preserves_input_and_dict_aliases():
    values = _assignment_store_kwargs()
    phase_screens = {}
    store = CanonicalAssignmentStore(
        **values,
        structure_phase_screens=phase_screens,
    )

    assert CanonicalAssignmentStore is AssignmentStore
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            stored = getattr(store, name)
            assert stored is value
            assert stored.untyped_storage().data_ptr() == (
                value.untyped_storage().data_ptr()
            )
    assert store.structure_phase_screens is phase_screens


def test_assignment_store_default_factory_is_independent_per_instance():
    first = CanonicalAssignmentStore(**_assignment_store_kwargs())
    second = CanonicalAssignmentStore(**_assignment_store_kwargs())

    assert first.structure_phase_screens == {}
    assert second.structure_phase_screens == {}
    assert first.structure_phase_screens is not second.structure_phase_screens


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    (
        (
            {
                "face_material_id": torch.zeros((2,), dtype=torch.float32),
                "edge_material_id0": torch.zeros((2,), dtype=torch.int32),
            },
            TypeError,
            "face_material_id must have dtype torch.int32",
        ),
        (
            {
                "face_material_id": torch.zeros((2,), dtype=torch.int32),
                "edge_material_id0": torch.zeros((2,), dtype=torch.int32),
            },
            ValueError,
            "face_material_id length must match num_faces",
        ),
        (
            {
                "edge_material_id0": torch.zeros((2,), dtype=torch.int32),
                "edge_material_id1": torch.zeros((2,), dtype=torch.int32),
            },
            ValueError,
            "edge_material_id0 length must match num_edges",
        ),
        (
            {"structure_phase_screens": {-1: object()}},
            ValueError,
            "structure_phase_screens keys must be structure indices",
        ),
        (
            {"structure_phase_screens": {0: object()}},
            ValueError,
            "structure_phase_screens values must be PhaseScreen",
        ),
    ),
)
def test_assignment_store_validation_order_and_errors_are_exact(
    changes, error, message
):
    values = _assignment_store_kwargs()
    values.update(changes)

    with pytest.raises(error, match=message):
        CanonicalAssignmentStore(**values)
