"""Compatibility imports for the canonical scene model owner."""

import hashlib  # noqa: F401
import json  # noqa: F401
from dataclasses import dataclass, field, replace  # noqa: F401

import torch  # noqa: F401

from witwin.channel_native.core.edge_policy import (  # noqa: F401
    DEFAULT_EDGE_POLICY,
    EdgePolicy,
)
from witwin.channel_native.core.edge_selection import (  # noqa: F401
    resolve_scene_edge_policy,
)
from witwin.channel_native.materials.models import (  # noqa: F401
    GEOMETRY_MODE_IDS,
    MATERIAL_ABI_VERSION,
    PhaseScreen,
    SurfaceAssignment,
    effective_sigma_e,
)
from witwin.channel_native.propagation.geometry.kernels import (  # noqa: F401
    primitives as geometry_primitives,
)
from witwin.channel_native.propagation.topology.kernels import (  # noqa: F401
    primitives as topology_primitives,
)
from witwin.channel_native.runtime.native_buffers import bdpt_zero_matrix  # noqa: F401
from witwin.channel_native.scene.compile import (  # noqa: F401
    _abi_v3_layer_view,
    _compile_assignments,
    _compile_geometry,
    _compile_materials,
    _frequency_dependent_material_keys,
    _material_records,
    _phase_screen_descriptor,
    compile_scene,
)
from witwin.channel_native.scene.compiled import CompiledScene  # noqa: F401
from witwin.channel_native.scene.kernels.rayd_scene import (  # noqa: F401
    RayDNScene,
    build_scene_from_structures,
)
from witwin.channel_native.scene.models import (  # noqa: F401
    Receiver,
    ReceiverGrid,
    ReceiverPoint,
    Scene,
    Structure,
    Transmitter,
    _diffraction_edge_count_from_raydn_scene,
    _RAYD_EDGE_INFO_PLANE_TOL,
)
from witwin.channel_native.scene.stores.assignments import AssignmentStore  # noqa: F401
from witwin.channel_native.scene.stores.geometry import GeometryStore  # noqa: F401
from witwin.channel_native.scene.stores.materials import MaterialStore  # noqa: F401
