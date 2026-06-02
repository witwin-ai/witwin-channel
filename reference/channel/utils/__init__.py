"""Utility exports for DrJit and torch interop."""

from .angles import spherical_angles
from .conversion import corner_xy, edge_xy, scalar, to_numpy, to_numpy_2d, to_numpy_complex_2d
from .drjit_ops import ArrayInit, Broadcast, Concat, EvalSync, Gather
from .mesh_buffers import faces_array, to_point3f, to_vector3u, vertices_array
from .power import to_power_db
from .tensor_conversion import (
    BoolTensor,
    FloatTensor,
    IntTensor,
    reshape_or_broadcast_torch_tensor,
    shape_tuple,
    to_bool_tensor,
    to_complex_array,
    to_float_tensor,
    to_int_tensor,
    to_mapping_proxy,
    to_typed_tensor,
    to_vector_tensor,
)
from .transform import Transform4f
from .torch_bridge import (
    drjit_add,
    drjit_forward_ad,
    drjit_mul,
    drjit_norm,
    drjit_sqrt,
    drjit_to_torch,
    drjit_to_torch_view,
    stable_torch_argsort,
    to_drjit,
    to_torch,
    torch_lexsort,
    torch_to_drjit,
    wrap_drjit,
    wrap_drjit_forward_ad,
)


__all__ = [
    "ArrayInit",
    "Broadcast",
    "Concat",
    "EvalSync",
    "Gather",
    "corner_xy",
    "drjit_add",
    "drjit_forward_ad",
    "drjit_mul",
    "drjit_norm",
    "drjit_sqrt",
    "drjit_to_torch",
    "drjit_to_torch_view",
    "edge_xy",
    "faces_array",
    "scalar",
    "spherical_angles",
    "stable_torch_argsort",
    "to_drjit",
    "to_numpy",
    "to_numpy_2d",
    "to_numpy_complex_2d",
    "to_point3f",
    "to_power_db",
    "to_torch",
    "to_vector3u",
    "torch_lexsort",
    "torch_to_drjit",
    "vertices_array",
    "Transform4f",
    "wrap_drjit",
    "wrap_drjit_forward_ad",
]

