"""Compatibility imports for the canonical material encoding owner."""

from typing import TYPE_CHECKING  # noqa: F401 - legacy reachable global

import torch  # noqa: F401 - legacy reachable global

from witwin.channel_native.materials.encoding import (  # noqa: F401
    face_material_field_bundle,
    face_material_tensors,
    face_material_thickness,
)
from witwin.channel_native.materials.kernels.functional import (  # noqa: F401
    mc_face_material_tensors,
)
from witwin.channel_native.materials.models import (  # noqa: F401
    PEC_EFFECTIVE_SIGMA_E,
    PEC_MODEL_ID,
)
