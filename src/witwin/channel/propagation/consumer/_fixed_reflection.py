"""Fixed-topology reevaluation for frozen rows that carry interactions.

A frozen reflection row is a face sequence, not a fixed point in space. At new
endpoint positions its stationary point has to be resolved again, because the
specular point moves and can leave its facet or become occluded. This module
replays exactly the owners the discovery path used - the RayD fixed-winner EPC
re-solve for the geometry, and the native reflection field transport for the
field - so a reevaluated row is the value discovery would have produced at
those endpoints.

Two things drive the shape of the code.

The native reflection transport takes ONE uniform interaction depth per launch,
so a mixed-depth frozen batch is replayed one ``(component, depth)`` bucket at a
time. Those buckets come from ``prepare_fixed_topology`` and are host-known, so
the per-call path never observes a device count to decide how to launch.

A frozen path can legitimately stop existing. Failing the whole batch would
force a caller back to full discovery the first time one path dies, which
defeats the capability, so validity is published per row. An invalid row is NOT
a failure: it is the correct, complete answer that this frozen path does not
exist at these endpoints. Capacity, ABI, contract, and device failures remain
all-or-nothing and still raise before a result exists.

An invalid row is made inert at the input, not patched at the output: its
transmit polarization is replaced by the zero vector, and the native transport
carries that exactly through projection, every Fresnel bounce, and the trailing
free-space scalar, so all four field outputs come out as exact zeros from the
kernel that owns them. Only the scalar path geometry, which has no such inert
excitation, is selected against the mask afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ._jones import compose_jones, transverse_basis
from ._rows import PreparedRows, select_rows
from .contracts import FixedTopologyBucket, PreparedFixedTopology

if TYPE_CHECKING:
    from witwin.channel.scene.compiled import CompiledScene


_MATERIAL_FIELDS = ("eps_r", "sigma_e", "mu_r", "gain", "thickness")


@dataclass(frozen=True, slots=True)
class BucketInputs:
    """Everything one bucket hands to the native transport."""

    depth: int
    source: torch.Tensor
    target: torch.Tensor
    tx_power: torch.Tensor
    tx_polarization: torch.Tensor
    rx_polarization: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    material: tuple[torch.Tensor, ...]
    valid: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.source.shape[0])


@dataclass(frozen=True, slots=True)
class FixedRowOutputs:
    """Frozen-order ``K`` row outputs of one reevaluation.

    ``path_field`` and ``path_field_vector`` are the source-excited transport,
    matching what discovery publishes. ``path_field_vector`` is only produced
    for the complex3 response, which is the only reader of it.
    """

    path_field: torch.Tensor
    path_field_vector: torch.Tensor | None
    path_length_m: torch.Tensor
    delay_s: torch.Tensor
    direction: torch.Tensor
    interaction_positions: torch.Tensor
    interaction_normals: torch.Tensor
    matrix: torch.Tensor | None
    source_basis: torch.Tensor | None
    sink_basis: torch.Tensor | None
    row_valid: torch.Tensor | None


def require_smooth_reflection_scene(compiled: CompiledScene) -> None:
    """Reject a scene whose reflection rows need rough-surface attenuation.

    The coherent rough-reflection factor and the realization phase-screen delta
    replacement are owned by the discovery-side field loop and are gated on
    host material state there. Reproducing that gate here would duplicate
    another owner's policy, and silently disagreeing with ``evaluate`` on a
    rough scene is worse than refusing, so this route requires a smooth scene.
    """

    if bool((compiled.materials.scatter_model_id == 1).any()):
        raise NotImplementedError(
            "fixed-topology reflection reevaluation requires a smooth scene; "
            "rough-surface coherent attenuation is owned by the discovery "
            "field loop and is not reproduced here"
        )
    screens = getattr(compiled.assignments, "structure_phase_screens", {})
    if any(
        getattr(screen, "mode", None) == "realization_coherent"
        for screen in screens.values()
    ):
        raise NotImplementedError(
            "fixed-topology reflection reevaluation does not support "
            "realization_coherent phase screens"
        )


def _scene_tables(compiled: CompiledScene) -> dict[str, object]:
    # CompiledScene owns the lazy scene-static cache (Plan-13 pattern); the
    # replay just consumes it. The primal per-frame case stages host-to-device
    # once per compiled scene instead of once per call.
    return compiled.fixed_reevaluation_tables()


def _reflection_inputs(
    compiled: CompiledScene,
    bucket: FixedTopologyBucket,
    tables: dict[str, object],
    prepared: PreparedFixedTopology,
    rows: PreparedRows,
) -> BucketInputs:
    from witwin.channel.propagation.geometry.reevaluate import (
        reflection_epc_paths,
    )

    depth = bucket.depth
    sequence = select_rows(
        prepared.topology.primitive_sequence, bucket.rows
    )[:, :depth].contiguous()
    face_id = sequence.to(dtype=torch.int64)
    source = select_rows(rows.source, bucket.rows)
    target = select_rows(rows.target, bucket.rows)
    epc = reflection_epc_paths(
        compiled, tables["vertices"], source, target, face_id, depth
    )
    valid = epc["valid"]
    material = tables["material"]
    return BucketInputs(
        depth=depth,
        source=source,
        target=target,
        tx_power=select_rows(rows.tx_power, bucket.rows),
        tx_polarization=_inert_where_invalid(
            select_rows(rows.tx_polarization, bucket.rows), valid
        ),
        rx_polarization=select_rows(rows.rx_polarization, bucket.rows),
        interaction_positions=epc["hit_positions"],
        interaction_normals=epc["normals"],
        material=tuple(material[name][face_id].contiguous() for name in _MATERIAL_FIELDS),
        valid=valid,
    )


def _los_inputs(
    compiled: CompiledScene, bucket: FixedTopologyBucket, rows: PreparedRows
) -> BucketInputs:
    source = select_rows(rows.source, bucket.rows)
    target = select_rows(rows.target, bucket.rows)
    empty = source.new_empty((int(source.shape[0]), 0, 3))
    # Re-test visibility with the same native gate discovery applies to LoS
    # candidates, so a sink that moved behind a wall publishes row_valid=False
    # and exact zeros instead of a full-strength free-space answer. A
    # structure-less scene cannot occlude anything and skips the launch.
    if compiled.structures:
        from witwin.channel.propagation.geometry.visibility import (
            VisibilityQuery,
            run_visibility_query,
        )

        valid = run_visibility_query(
            VisibilityQuery(
                rayd=compiled.rayd,
                start=source.contiguous(),
                end=target.contiguous(),
                active=None,
            )
        ).visible
    else:
        valid = torch.ones(
            (int(source.shape[0]),), dtype=torch.bool, device=source.device
        )
    return BucketInputs(
        depth=0,
        source=source,
        target=target,
        tx_power=select_rows(rows.tx_power, bucket.rows),
        tx_polarization=_inert_where_invalid(
            select_rows(rows.tx_polarization, bucket.rows), valid
        ),
        rx_polarization=select_rows(rows.rx_polarization, bucket.rows),
        interaction_positions=empty,
        interaction_normals=empty,
        material=(),
        valid=valid,
    )


def _inert_where_invalid(
    values: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Select a row's value or the inert constant; never a numerical blend."""

    shape = (-1, *((1,) * (values.ndim - 1)))
    return torch.where(valid.reshape(shape), values, values.new_zeros(()))


def _field_op(
    inputs: BucketInputs,
    *,
    ad_mode: str,
    frequency: float | torch.Tensor,
    frequency_value: float,
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    from witwin.channel.propagation.fields.kernels import (
        autograd as field_autograd,
    )
    from witwin.channel.propagation.fields.kernels import (
        functional as field_functional,
    )

    differentiable = ad_mode != "none"

    def run(
        tx_polarization: torch.Tensor, rx_polarization: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if inputs.depth == 0:
            leading = (inputs.source, inputs.target)
            trailing: tuple[torch.Tensor, ...] = ()
        else:
            leading = (
                inputs.source,
                inputs.target,
                inputs.interaction_positions,
                inputs.interaction_normals,
            )
            trailing = inputs.material
        arguments = (
            *leading,
            inputs.tx_power,
            tx_polarization,
            rx_polarization,
            *trailing,
        )
        if differentiable:
            operator = (
                field_autograd.field_free_space_ad
                if inputs.depth == 0
                else field_autograd.field_reflection_sequence_ad
            )
            return operator(
                *arguments, frequency=frequency, frequency_value=frequency_value
            )
        operator = (
            field_functional.field_free_space
            if inputs.depth == 0
            else field_functional.field_reflection_sequence
        )
        return operator(*arguments, frequency_hz=frequency_value)

    return run


def _leg_endpoints(inputs: BucketInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the first leg ends and where the last leg starts.

    A reflection row launches toward its first interaction and arrives from
    its last one, so its transverse bases live in two different planes. Both
    are read off the interaction table the native transport itself consumes;
    neither direction is recomputed here.
    """

    if inputs.depth == 0:
        return inputs.target, inputs.source
    return (
        inputs.interaction_positions[:, 0].contiguous(),
        inputs.interaction_positions[:, inputs.depth - 1].contiguous(),
    )


def _jones_values(
    inputs: BucketInputs,
    run: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    source_reference: torch.Tensor,
    sink_reference: torch.Tensor,
    frequency_value: float,
) -> dict[str, torch.Tensor]:
    launch_target, arrival_origin = _leg_endpoints(inputs)
    source_basis = _inert_where_invalid(
        transverse_basis(
            source_reference,
            inputs.source,
            launch_target,
            frequency_hz=frequency_value,
        ),
        inputs.valid,
    )
    matrix, sink_basis, column = compose_jones(
        lambda polarization: run(polarization, polarization),
        source_basis=source_basis,
        sink_reference_basis=sink_reference,
        arrival_origin=arrival_origin,
        arrival_target=inputs.target,
        frequency_hz=frequency_value,
    )
    return {
        **column,
        "matrix": matrix,
        "source_basis": source_basis,
        "sink_basis": _inert_where_invalid(sink_basis, inputs.valid),
    }


def _bucket_values(
    inputs: BucketInputs,
    run: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    response: str,
    ad_mode: str,
    source_reference: torch.Tensor | None,
    sink_reference: torch.Tensor | None,
    frequency_value: float,
) -> dict[str, torch.Tensor]:
    if response != "polarimetric_transport":
        values = run(inputs.tx_polarization, inputs.rx_polarization)
    else:
        assert source_reference is not None and sink_reference is not None
        values = _jones_values(
            inputs,
            run,
            source_reference=source_reference,
            sink_reference=sink_reference,
            frequency_value=frequency_value,
        )
    if response != "complex3_transport":
        return values
    from ._amplitude import excited_field

    return {
        **values,
        "path_field_vector": excited_field(
            values["field_vector"], inputs.tx_power, ad_mode=ad_mode
        ),
    }


def _pad_interactions(values: torch.Tensor, width: int) -> torch.Tensor:
    depth = int(values.shape[1])
    if depth == width:
        return values
    padding = values.new_zeros((int(values.shape[0]), width - depth, 3))
    return torch.cat((values, padding), dim=1)


def _publish_bucket(
    outputs: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    inputs: BucketInputs,
    bucket: FixedTopologyBucket,
    width: int,
) -> None:
    rows = bucket.rows
    valid = inputs.valid
    outputs["path_field"].index_copy_(0, rows, values["path_field"])
    if outputs["path_field_vector"] is not None:
        outputs["path_field_vector"].index_copy_(
            0, rows, values["path_field_vector"]
        )
    outputs["path_length_m"].index_copy_(
        0, rows, _inert_where_invalid(values["path_length_m"], valid)
    )
    outputs["delay_s"].index_copy_(
        0, rows, _inert_where_invalid(values["delay_s"], valid)
    )
    outputs["direction"].index_copy_(
        0, rows, _inert_where_invalid(values["direction"], valid)
    )
    outputs["interaction_positions"].index_copy_(
        0, rows, _pad_interactions(inputs.interaction_positions, width)
    )
    outputs["interaction_normals"].index_copy_(
        0, rows, _pad_interactions(inputs.interaction_normals, width)
    )
    outputs["row_valid"].index_copy_(0, rows, valid)
    for name in ("matrix", "source_basis", "sink_basis"):
        if outputs[name] is not None:
            outputs[name].index_copy_(0, rows, values[name])


def _allocate(
    rows: PreparedRows, width: int, response: str
) -> dict[str, torch.Tensor | None]:
    count = rows.row_count
    device = rows.source.device
    polarimetric = response == "polarimetric_transport"
    return {
        "path_field": torch.zeros(
            (count,), dtype=torch.complex64, device=device
        ),
        "path_field_vector": (
            torch.zeros((count, 3), dtype=torch.complex64, device=device)
            if response == "complex3_transport"
            else None
        ),
        "path_length_m": torch.zeros((count,), device=device),
        "delay_s": torch.zeros((count,), device=device),
        "direction": torch.zeros((count, 3), device=device),
        "interaction_positions": torch.zeros((count, width, 3), device=device),
        "interaction_normals": torch.zeros((count, width, 3), device=device),
        "row_valid": torch.ones((count,), dtype=torch.bool, device=device),
        "matrix": (
            torch.zeros((count, 2, 2), dtype=torch.complex64, device=device)
            if polarimetric
            else None
        ),
        "source_basis": (
            torch.zeros((count, 2, 3), device=device) if polarimetric else None
        ),
        "sink_basis": (
            torch.zeros((count, 2, 3), device=device) if polarimetric else None
        ),
    }


def evaluate_prepared(
    compiled: CompiledScene,
    prepared: PreparedFixedTopology,
    rows: PreparedRows,
    *,
    response: str,
    ad_mode: str,
    frequency: float | torch.Tensor,
    frequency_value: float,
    source_reference_basis: torch.Tensor | None,
    sink_reference_basis: torch.Tensor | None,
    publish_row_validity: bool,
) -> FixedRowOutputs:
    """Replay every host-known bucket of a prepared frozen topology."""

    width = int(prepared.topology.primitive_sequence.shape[1])
    outputs = _allocate(rows, width, response)
    tables = (
        _scene_tables(compiled)
        if any(bucket.depth > 0 for bucket in prepared.buckets)
        else {}
    )
    for bucket in prepared.buckets:
        inputs = (
            _los_inputs(compiled, bucket, rows)
            if bucket.depth == 0
            else _reflection_inputs(compiled, bucket, tables, prepared, rows)
        )
        values = _bucket_values(
            inputs,
            _field_op(
                inputs,
                ad_mode=ad_mode,
                frequency=frequency,
                frequency_value=frequency_value,
            ),
            response=response,
            ad_mode=ad_mode,
            source_reference=(
                None
                if source_reference_basis is None
                else select_rows(source_reference_basis, bucket.rows)
            ),
            sink_reference=(
                None
                if sink_reference_basis is None
                else select_rows(sink_reference_basis, bucket.rows)
            ),
            frequency_value=frequency_value,
        )
        _publish_bucket(outputs, values, inputs, bucket, width)
    return FixedRowOutputs(
        path_field=outputs["path_field"],
        path_field_vector=outputs["path_field_vector"],
        path_length_m=outputs["path_length_m"],
        delay_s=outputs["delay_s"],
        direction=outputs["direction"],
        interaction_positions=outputs["interaction_positions"],
        interaction_normals=outputs["interaction_normals"],
        matrix=outputs["matrix"],
        source_basis=outputs["source_basis"],
        sink_basis=outputs["sink_basis"],
        row_valid=outputs["row_valid"] if publish_row_validity else None,
    )


__all__ = [
    "BucketInputs",
    "FixedRowOutputs",
    "evaluate_prepared",
    "require_smooth_reflection_scene",
]
