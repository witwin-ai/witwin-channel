"""Compatibility imports for scene tensor projections.

The canonical implementation lives in :mod:`witwin.channel.scene.tensors`.
This module intentionally has no ``__all__`` so legacy wildcard imports keep
their historical behavior.
"""

from witwin.channel.scene.tensors import (  # noqa: F401
    LIGHT_SPEED_M_PER_S,
    ReceiverGrid,
    ReceiverPoint,
    TYPE_CHECKING,
    _frequency_scalar,
    host_vec3_tensor,
    mc_receiver_grid_points,
    mc_transmitter_tensors,
    receiver_grid_points,
    receiver_positions,
    topology_primitives,
    torch,
    transmitter_positions,
    vector3_tuple,
)
