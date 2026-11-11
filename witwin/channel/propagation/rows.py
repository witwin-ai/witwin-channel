# Copyright Xingyu Chen.
# Row-aligned propagation contracts shared by the propagation stages.

"""Row-aligned propagation contracts shared by the propagation stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from witwin.channel.runtime import require_tensor


@dataclass(frozen=True, slots=True, eq=False)
class _RowIdentity:
    """Opaque row token shared by all views of one path table."""

    row_count: int
    sequence_width: int
    device: torch.device


@dataclass(frozen=True, slots=True, eq=False)
class PathTopology:
    """Discrete path rows without continuous geometry, fields, or runtime state."""

    valid: torch.Tensor
    tx_id: torch.Tensor
    rx_id: torch.Tensor
    depth: torch.Tensor
    component_id: torch.Tensor
    primitive_id: torch.Tensor
    edge_id: torch.Tensor
    material_id: torch.Tensor
    primitive_sequence: torch.Tensor
    material_sequence: torch.Tensor
    interaction_type: torch.Tensor
    _row_identity: _RowIdentity = field(init=False, repr=False)

    def __post_init__(self) -> None:
        valid = require_tensor("valid", self.valid, dtype=torch.bool, ndim=1)
        row_count = int(valid.shape[0])
        device = valid.device
        vector_shape = (row_count,)
        for name in (
            "tx_id",
            "rx_id",
            "depth",
            "component_id",
            "primitive_id",
            "edge_id",
            "material_id",
        ):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=vector_shape,
                device=device,
            )

        primitive_sequence = require_tensor(
            "primitive_sequence",
            self.primitive_sequence,
            dtype=torch.int32,
            ndim=2,
            device=device,
        )
        sequence_width = int(primitive_sequence.shape[1])
        sequence_shape = (row_count, sequence_width)
        if tuple(primitive_sequence.shape) != sequence_shape:
            raise ValueError(
                "primitive_sequence must have shape "
                f"{sequence_shape}, got {tuple(primitive_sequence.shape)}"
            )
        for name in ("material_sequence", "interaction_type"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.int32,
                shape=sequence_shape,
                device=device,
            )

        object.__setattr__(
            self,
            "_row_identity",
            _RowIdentity(
                row_count=row_count,
                sequence_width=sequence_width,
                device=device,
            ),
        )

    @property
    def row_identity(self) -> _RowIdentity:
        """Opaque identity token; downstream contracts must reuse this object."""

        return self._row_identity

    @property
    def row_count(self) -> int:
        return self._row_identity.row_count

    @property
    def sequence_width(self) -> int:
        return self._row_identity.sequence_width

    @property
    def device(self) -> torch.device:
        return self._row_identity.device


@dataclass(frozen=True, slots=True, eq=False)
class PathGeometry:
    """Continuous geometry aligned to an existing ``PathTopology`` row table."""

    row_identity: _RowIdentity
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    field_direction: torch.Tensor
    interaction_position: torch.Tensor
    interaction_normal: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.row_identity, _RowIdentity):
            raise TypeError("row_identity must come from PathTopology")
        rows = self.row_identity.row_count
        width = self.row_identity.sequence_width
        device = self.row_identity.device
        for name in ("path_length_m", "delay_s"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows,),
                device=device,
            )
        for name in (
            "field_direction",
            "interaction_position",
            "interaction_normal",
        ):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows, 3),
                device=device,
            )
        for name in ("interaction_positions", "interaction_normals"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.float32,
                shape=(rows, width, 3),
                device=device,
            )

    @property
    def row_count(self) -> int:
        return self.row_identity.row_count

    @property
    def device(self) -> torch.device:
        return self.row_identity.device


@dataclass(frozen=True, slots=True, eq=False)
class PathFields:
    """RF fields aligned to an existing ``PathTopology`` row table."""

    row_identity: _RowIdentity
    path_gain: torch.Tensor
    path_field: torch.Tensor
    field_xyz: torch.Tensor
    coefficient: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.row_identity, _RowIdentity):
            raise TypeError("row_identity must come from PathTopology")
        rows = self.row_identity.row_count
        device = self.row_identity.device
        require_tensor(
            "path_gain",
            self.path_gain,
            dtype=torch.float32,
            shape=(rows,),
            device=device,
        )
        for name in ("path_field", "coefficient"):
            require_tensor(
                name,
                getattr(self, name),
                dtype=torch.complex64,
                shape=(rows,),
                device=device,
            )
        require_tensor(
            "field_xyz",
            self.field_xyz,
            dtype=torch.complex64,
            shape=(rows, 3),
            device=device,
        )

    @property
    def row_count(self) -> int:
        return self.row_identity.row_count

    @property
    def device(self) -> torch.device:
        return self.row_identity.device


@dataclass(frozen=True, slots=True, eq=False)
class EvaluatedPaths:
    """Internal propagation result with exact shared path-row identity."""

    topology: PathTopology
    geometry: PathGeometry
    fields: PathFields

    def __post_init__(self) -> None:
        if not isinstance(self.topology, PathTopology):
            raise TypeError("topology must be a PathTopology")
        if not isinstance(self.geometry, PathGeometry):
            raise TypeError("geometry must be a PathGeometry")
        if not isinstance(self.fields, PathFields):
            raise TypeError("fields must be PathFields")
        identity = self.topology.row_identity
        if self.geometry.row_identity is not identity:
            raise ValueError("geometry must share topology row_identity")
        if self.fields.row_identity is not identity:
            raise ValueError("fields must share topology row_identity")

    @property
    def row_identity(self) -> _RowIdentity:
        return self.topology.row_identity

    @property
    def row_count(self) -> int:
        return self.topology.row_count

    @property
    def device(self) -> torch.device:
        return self.topology.device


COMPONENT_ROW_GROUPS = 8


@dataclass(frozen=True, slots=True, eq=False)
class PathRowGroups:
    """Host row counts and a device order that list each field-stage group's rows.

    Group ``c`` holds the rows of component ``c`` (reflection excepted), group
    ``8 + d`` the reflection rows of depth ``d``, and the last group every other row. Rows of
    one group appear in ascending row order inside ``order``. ``order`` is None when the rows
    are already grouped, in which case every row belongs to the single non-empty group.
    """

    order: torch.Tensor | None
    counts: tuple[int, ...]
    offsets: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        offsets = [0]
        for value in self.counts:
            if type(value) is not int or value < 0:
                raise ValueError("row group counts must be non-negative ints")
            offsets.append(offsets[-1] + value)
        object.__setattr__(self, "offsets", tuple(offsets))
        if self.order is not None:
            require_tensor("order", self.order, dtype=torch.int64, shape=(offsets[-1],))
        elif sum(1 for value in self.counts if value) > 1:
            raise ValueError("ungrouped rows must belong to a single group")

    @classmethod
    def single(cls, group: int, row_count: int, *, sequence_width: int) -> PathRowGroups:
        counts = [0] * (COMPONENT_ROW_GROUPS + int(sequence_width) + 2)
        counts[group] = int(row_count)
        return cls(order=None, counts=tuple(counts))

    @property
    def row_count(self) -> int:
        return self.offsets[-1]

    def count(self, group: int) -> int:
        return self.counts[group] if 0 <= group < len(self.counts) else 0

    def rows(self, group: int) -> torch.Tensor | None:
        """Row indices of one non-empty group, or None when the group holds every row."""

        count = self.count(group)
        if count == 0:
            raise ValueError(f"row group {group} is empty")
        if count == self.row_count:
            return None
        assert self.order is not None
        start = self.offsets[group]
        return self.order[start : start + count]

    def reflection_depths(self) -> tuple[int, ...]:
        """Reflection depths that have rows, ascending."""

        width = len(self.counts) - COMPONENT_ROW_GROUPS - 1
        return tuple(
            depth for depth in range(1, width) if self.count(COMPONENT_ROW_GROUPS + depth)
        )


__all__ = ["EvaluatedPaths", "PathFields", "PathGeometry", "PathRowGroups", "PathTopology"]
