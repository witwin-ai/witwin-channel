from __future__ import annotations

import torch

from witwin.channel_native.propagation.topology.kernels import (
    construction as topology_construction,
)


def _ensure_topology_fields(
    block: dict[str, torch.Tensor],
    *,
    interaction_position: torch.Tensor | None = None,
    interaction_normal: torch.Tensor | None = None,
    material_id: torch.Tensor | None = None,
    path_field: torch.Tensor | None = None,
    field_xyz: torch.Tensor | None = None,
    coefficient: torch.Tensor | None = None,
    primitive_sequence: torch.Tensor | None = None,
    material_sequence: torch.Tensor | None = None,
    interaction_positions: torch.Tensor | None = None,
    interaction_normals: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    extended = dict(block)
    defaults: dict[str, torch.Tensor] | None = None

    def topology_defaults() -> dict[str, torch.Tensor]:
        nonlocal defaults
        if defaults is None:
            defaults = topology_construction.deterministic_topology_default_fields(
                block["path_gain"].to(dtype=torch.float32).contiguous()
            )
        return defaults

    interaction_position_value = (
        interaction_position
        if interaction_position is not None
        else block.get("interaction_position")
    )
    if interaction_position_value is None:
        interaction_position_value = topology_defaults()["interaction_position"]
    interaction_normal_value = (
        interaction_normal
        if interaction_normal is not None
        else block.get("interaction_normal")
    )
    if interaction_normal_value is None:
        interaction_normal_value = topology_defaults()["interaction_normal"]
    material_id_value = (
        material_id if material_id is not None else block.get("material_id")
    )
    if material_id_value is None:
        material_id_value = topology_defaults()["material_id"]
    path_field_value = path_field if path_field is not None else block.get("path_field")
    if path_field_value is None:
        path_field_value = topology_defaults()["path_field"]
    field_xyz_value = field_xyz if field_xyz is not None else block.get("field_xyz")
    if field_xyz_value is None:
        field_xyz_value = torch.zeros(
            (int(block["valid"].shape[0]), 3),
            device=block["valid"].device,
            dtype=torch.complex64,
        )
    coefficient_value = (
        coefficient if coefficient is not None else block.get("coefficient")
    )
    if coefficient_value is None:
        coefficient_value = path_field_value
    extended["interaction_position"] = interaction_position_value.to(
        dtype=torch.float32
    ).contiguous()
    extended["interaction_normal"] = interaction_normal_value.to(
        dtype=torch.float32
    ).contiguous()
    extended["material_id"] = material_id_value.to(dtype=torch.int32).contiguous()
    extended["path_field"] = path_field_value.to(dtype=torch.complex64).contiguous()
    extended["field_xyz"] = field_xyz_value.to(dtype=torch.complex64).contiguous()
    extended["coefficient"] = coefficient_value.to(dtype=torch.complex64).contiguous()
    if primitive_sequence is not None:
        extended["primitive_sequence"] = primitive_sequence.to(
            dtype=torch.int32
        ).contiguous()
    if material_sequence is not None:
        extended["material_sequence"] = material_sequence.to(
            dtype=torch.int32
        ).contiguous()
    if interaction_positions is not None:
        extended["interaction_positions"] = interaction_positions.to(
            dtype=torch.float32
        ).contiguous()
    if interaction_normals is not None:
        extended["interaction_normals"] = interaction_normals.to(
            dtype=torch.float32
        ).contiguous()
    return extended


__all__: list[str] = []
