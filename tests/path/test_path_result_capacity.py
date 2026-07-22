from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from witwin.channel.path import capacity as capacity_ops
from witwin.channel.path.capacity import from_capacity_evaluated_paths
from witwin.channel.path.result import InteractionType, endpoint_angles
from witwin.channel.propagation.enumerated.capacity import (
    evaluated_paths_capacity_pack,
)
from witwin.channel.propagation.models.capacity import CapacityEvaluatedPaths
from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.models.fields import PathFields
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology
from witwin.channel.runtime.capacity import create_capacity_failure_state
from witwin.channel.runtime import symbols


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _candidate_paths(*, differentiable: bool = False) -> EvaluatedPaths:
    valid = torch.tensor(
        [True, False, True, True, True, True], device="cuda", dtype=torch.bool
    )
    tx_id = torch.tensor([1, -2**31, 0, 1, 0, 1], device="cuda", dtype=torch.int32)
    rx_id = torch.tensor([1, -2**31, 0, 1, 0, 0], device="cuda", dtype=torch.int32)
    depth = torch.tensor([1, 2, 0, 2, 1, 1], device="cuda", dtype=torch.int32)
    component_id = torch.tensor([2, -1, 0, 3, 1, 5], device="cuda", dtype=torch.int32)
    edge_id = torch.tensor([41, -1, -1, 42, -1, -1], device="cuda", dtype=torch.int32)
    primitive_sequence = torch.tensor(
        [[-1, -1], [-1, -1], [-1, -1], [13, 14], [15, -1], [16, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    material_sequence = torch.tensor(
        [[3, -1], [-1, -1], [-1, -1], [4, 5], [6, -1], [7, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    interaction_type = torch.tensor(
        [
            [InteractionType.DIFFRACTION, InteractionType.NONE],
            [InteractionType.DIFFRACTION, InteractionType.DIFFRACTION],
            [InteractionType.NONE, InteractionType.NONE],
            [InteractionType.REFLECTION, InteractionType.DIFFRACTION],
            [InteractionType.REFLECTION, InteractionType.NONE],
            [InteractionType.TRANSMISSION, InteractionType.NONE],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    rows = torch.arange(6, device="cuda", dtype=torch.float32)
    first = torch.stack((rows + 1.0, rows + 2.0, rows + 3.0), dim=-1)
    second = torch.stack((rows + 4.0, rows + 5.0, rows + 6.0), dim=-1)
    positions = torch.stack((first, second), dim=1)
    normals = torch.stack((first * 0.1, second * 0.1), dim=1)
    normals[0, 0, 0] = float("nan")
    normals[3, 1, 2] = float("inf")
    direction = torch.stack((rows + 0.2, rows + 0.4, rows + 0.8), dim=-1)
    field_xyz = torch.complex(direction + 2.0, direction + 3.0)
    coefficient = torch.complex(rows + 4.0, -(rows + 5.0))
    for tensor in (positions, normals, direction, field_xyz, coefficient):
        poison = (
            complex(float("nan"), float("nan"))
            if tensor.is_complex()
            else float("nan")
        )
        tensor[1] = poison
    topology = PathTopology(
        valid=valid,
        tx_id=tx_id,
        rx_id=rx_id,
        depth=depth,
        component_id=component_id,
        primitive_id=primitive_sequence[:, 0].contiguous(),
        edge_id=edge_id,
        material_id=material_sequence[:, 0].contiguous(),
        primitive_sequence=primitive_sequence,
        material_sequence=material_sequence,
        interaction_type=interaction_type,
    )

    def live(value: torch.Tensor) -> torch.Tensor:
        return value.contiguous().requires_grad_(differentiable)

    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=live(rows + 10.0),
        delay_s=live((rows + 1.0) * 1.0e-9),
        field_direction=live(direction),
        interaction_position=live(first),
        interaction_normal=live(normals[:, 0].clone()),
        interaction_positions=live(positions),
        interaction_normals=live(normals),
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=live(rows + 20.0),
        path_field=live(coefficient * 2.0),
        field_xyz=live(field_xyz),
        coefficient=live(coefficient),
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


def _packed(*, capacity: int = 3, differentiable: bool = False):
    candidates = _candidate_paths(differentiable=differentiable)
    state = create_capacity_failure_state(candidates.topology.valid)
    packed = evaluated_paths_capacity_pack(
        candidates,
        failure_state=state,
        pair_count=4,
        num_tx=2,
        num_rx=2,
        path_capacity_per_pair=capacity,
    )
    return candidates, packed, state


def _endpoints(*, differentiable: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    tx = torch.tensor(
        [[-0.0, 0.0, 0.0], [10.0, 2.0, -3.0]],
        device="cuda",
        dtype=torch.float32,
        requires_grad=differentiable,
    )
    rx = torch.tensor(
        [[20.0, 7.0, 1.0], [30.0, -4.0, 9.0]],
        device="cuda",
        dtype=torch.float32,
        requires_grad=differentiable,
    )
    return tx, rx


def _result(packed: CapacityEvaluatedPaths, *, differentiable: bool = False):
    tx, rx = _endpoints(differentiable=differentiable)
    return (
        from_capacity_evaluated_paths(
            packed,
            num_rx=2,
            num_tx=2,
            tx_positions=tx,
            rx_positions=rx,
            metadata={"fixture": "capacity"},
        ),
        tx,
        rx,
    )


def _flat(result, name: str) -> torch.Tensor:
    value = getattr(result, name)
    tail = value.shape[5:]
    return value.reshape(12, *tail)


def _assert_positive_zero(value: torch.Tensor) -> None:
    assert torch.isfinite(value).all()
    assert torch.count_nonzero(value).item() == 0
    if value.is_complex():
        assert not torch.signbit(value.real).any()
        assert not torch.signbit(value.imag).any()
    elif value.is_floating_point():
        assert not torch.signbit(value).any()


def _assert_canonical_inert(result, *, capacity: int) -> None:
    """Check all 24 materialized and public derived inert members."""

    for value in (
        result.a,
        result.theta_t,
        result.phi_t,
        result.theta_r,
        result.phi_r,
        result.position,
        result.normal,
        result.field_xyz,
        result.field_direction,
    ):
        _assert_positive_zero(value)
    assert torch.all(result.tau == -1.0)
    assert not torch.any(result.valid)
    assert torch.all(result.interaction_type == 0)
    assert torch.all(result.primitive_id == -1)
    assert torch.all(result.material_id == -1)
    assert torch.count_nonzero(result.num_paths).item() == 0

    # Five derived inert views, two optional members, and the two public
    # capacity-shape members complete the 24-member PathResult contract view.
    assert torch.all(result.path_length_m == -1.0)
    assert result.types is result.interaction_type
    assert result.vertices is result.position
    assert result.normals is result.normal
    assert result.objects is result.primitive_id
    assert result.tx_weights is None
    assert result.rx_weights is None
    assert result.max_num_paths == capacity
    assert result.path_shape[4] == capacity


def _fill_numeric_poison(value: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    poison: int | float | complex
    if value.is_complex():
        poison = complex(float("nan"), float("inf"))
    elif value.is_floating_point():
        poison = float("nan")
    else:
        poison = torch.iinfo(value.dtype).min
    if mask is None:
        value.fill_(poison)
    else:
        value[mask] = poison


def _poison_invalid_capacity_rows(packed: CapacityEvaluatedPaths) -> None:
    """Poison capacity rows after selection, immediately before result packing."""

    topology = packed.evaluated.topology
    geometry = packed.evaluated.geometry
    fields = packed.evaluated.fields
    invalid = ~topology.valid
    with torch.no_grad():
        for value in (
            topology.tx_id,
            topology.rx_id,
            topology.depth,
            topology.component_id,
            topology.primitive_id,
            topology.edge_id,
            topology.material_id,
            topology.primitive_sequence,
            topology.material_sequence,
            topology.interaction_type,
            geometry.path_length_m,
            geometry.delay_s,
            geometry.field_direction,
            geometry.interaction_position,
            geometry.interaction_normal,
            geometry.interaction_positions,
            geometry.interaction_normals,
            fields.path_gain,
            fields.path_field,
            fields.field_xyz,
            fields.coefficient,
        ):
            _fill_numeric_poison(value, invalid)


def _poison_all_capacity_pack_inputs(packed: CapacityEvaluatedPaths) -> None:
    """Make every non-control pack input unsafe to read after a failure gate."""

    topology = packed.evaluated.topology
    geometry = packed.evaluated.geometry
    fields = packed.evaluated.fields
    with torch.no_grad():
        topology.valid.fill_(True)
        packed.selection.layout.num_paths.fill_(torch.iinfo(torch.int32).max)
        for value in (
            topology.tx_id,
            topology.rx_id,
            topology.depth,
            topology.component_id,
            topology.primitive_id,
            topology.edge_id,
            topology.material_id,
            topology.primitive_sequence,
            topology.material_sequence,
            topology.interaction_type,
            geometry.path_length_m,
            geometry.delay_s,
            geometry.field_direction,
            geometry.interaction_position,
            geometry.interaction_normal,
            geometry.interaction_positions,
            geometry.interaction_normals,
            fields.path_gain,
            fields.path_field,
            fields.field_xyz,
            fields.coefficient,
        ):
            _fill_numeric_poison(value)


def _poisoned_endpoints(*, differentiable: bool = False):
    tx = torch.full((2, 3), float("nan"), device="cuda")
    rx = torch.full((2, 3), float("inf"), device="cuda")
    return tx.requires_grad_(differentiable), rx.requires_grad_(differentiable)


def _assert_float_branch_parity(
    actual: torch.Tensor, expected: torch.Tensor
) -> None:
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))


def test_path_result_capacity_missing_symbol_fails_without_fallback(
    monkeypatch,
) -> None:
    assert capacity_ops._required_native_op is symbols.required_symbol
    _, packed, _ = _packed()
    tx, rx = _endpoints()
    requests: list[str] = []

    def missing(name: str):
        requests.append(name)
        raise symbols.NativeSymbolError(
            "required path result capacity symbol is missing"
        )

    monkeypatch.setattr(capacity_ops, "_required_native_op", missing)
    with pytest.raises(
        symbols.NativeSymbolError,
        match="required path result capacity symbol is missing",
    ):
        capacity_ops.from_capacity_evaluated_paths(
            packed,
            num_rx=2,
            num_tx=2,
            tx_positions=tx,
            rx_positions=rx,
        )
    assert requests == ["path_result_capacity_pack"]


def test_path_result_capacity_is_pair_major_exact_and_inert() -> None:
    _, packed, state = _packed()
    _poison_invalid_capacity_rows(packed)
    result, tx_positions, rx_positions = _result(packed)
    assert packed.selection.layout.failure_state is state
    assert result.a.shape == (2, 1, 2, 1, 3, 1)
    assert result.max_num_paths == 3
    assert result.num_paths.shape == (2, 1, 2, 1)
    assert result.num_paths.reshape(-1).tolist() == [2, 1, 0, 2]
    assert result.valid.reshape(-1).tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert result.metadata["fixture"] == "capacity"
    assert result.metadata["path_capacity_per_pair"] == 3

    topology = packed.evaluated.topology
    geometry = packed.evaluated.geometry
    fields = packed.evaluated.fields
    valid = topology.valid
    tx = tx_positions[topology.tx_id[valid].to(torch.int64)]
    rx = rx_positions[topology.rx_id[valid].to(torch.int64)]
    direct = rx - tx
    first = geometry.interaction_positions[valid, 0]
    last_slot = topology.depth[valid].to(torch.int64).clamp(min=1) - 1
    selected_positions = geometry.interaction_positions[valid]
    last = selected_positions[
        torch.arange(selected_positions.shape[0], device="cuda"), last_slot
    ]
    has_interaction = topology.depth[valid] > 0
    departure = torch.where(has_interaction[:, None], first - tx, direct)
    arrival = torch.where(has_interaction[:, None], rx - last, direct)
    theta_t, phi_t = endpoint_angles(departure)
    theta_r, phi_r = endpoint_angles(-arrival)
    for name, expected in (
        ("theta_t", theta_t),
        ("phi_t", phi_t),
        ("theta_r", theta_r),
        ("phi_r", phi_r),
    ):
        actual = _flat(result, name)[valid]
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(_flat(result, "tau")[valid], geometry.delay_s[valid])
    torch.testing.assert_close(
        _flat(result, "a")[valid, 0], fields.coefficient[valid]
    )
    torch.testing.assert_close(
        _flat(result, "field_xyz")[valid], fields.field_xyz[valid]
    )
    torch.testing.assert_close(
        _flat(result, "field_direction")[valid], geometry.field_direction[valid]
    )
    assert _flat(result, "primitive_id")[9, 0].item() == 41
    assert _flat(result, "normal")[9, 0, 0].item() == 0.0
    assert _flat(result, "normal")[10, 1, 2].item() == 0.0

    invalid = ~valid
    assert torch.count_nonzero(_flat(result, "a")[invalid]).item() == 0
    assert torch.all(_flat(result, "tau")[invalid] == -1.0)
    for name in ("theta_t", "phi_t", "theta_r", "phi_r"):
        assert torch.count_nonzero(_flat(result, name)[invalid]).item() == 0
    assert torch.all(_flat(result, "interaction_type")[invalid] == 0)
    assert torch.all(_flat(result, "primitive_id")[invalid] == -1)
    assert torch.all(_flat(result, "material_id")[invalid] == -1)
    for name in ("position", "normal", "field_xyz", "field_direction"):
        assert torch.count_nonzero(_flat(result, name)[invalid]).item() == 0


@pytest.mark.parametrize("failure_source", ("state", "overflow"))
def test_path_result_capacity_failure_is_completely_inert(failure_source: str) -> None:
    capacity = 1 if failure_source == "overflow" else 3
    _, packed, state = _packed(capacity=capacity)
    if failure_source == "overflow":
        assert packed.selection.layout.overflow.tolist() == [True]
        assert state.bits.item() != 0
    _poison_all_capacity_pack_inputs(packed)
    if failure_source == "state":
        state.bits.fill_(1)
    tx, rx = _poisoned_endpoints()
    result = from_capacity_evaluated_paths(
        packed,
        num_rx=2,
        num_tx=2,
        tx_positions=tx,
        rx_positions=rx,
    )
    _assert_canonical_inert(result, capacity=capacity)


@pytest.mark.parametrize("failure_source", ("state", "overflow"))
def test_path_result_capacity_failure_has_zero_vjp_and_jvp_on_stream(
    failure_source: str,
) -> None:
    capacity = 1 if failure_source == "overflow" else 3
    _, packed, state = _packed(capacity=capacity)
    if failure_source == "overflow":
        assert packed.selection.layout.overflow.tolist() == [True]
        assert state.bits.item() != 0
    _poison_all_capacity_pack_inputs(packed)
    if failure_source == "state":
        state.bits.fill_(1)
    differentiable, leaves = _clone_continuous_capacity(packed)
    tx, rx = _poisoned_endpoints(differentiable=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = from_capacity_evaluated_paths(
            differentiable,
            num_rx=2,
            num_tx=2,
            tx_positions=tx,
            rx_positions=rx,
        )
        loss = (
            result.a.real.sum()
            + result.a.imag.sum()
            + result.tau.sum()
            + result.theta_t.sum()
            + result.phi_t.sum()
            + result.theta_r.sum()
            + result.phi_r.sum()
            + result.position.sum()
            + result.normal.sum()
            + result.field_xyz.real.sum()
            + result.field_xyz.imag.sum()
            + result.field_direction.sum()
        )
        gradients = torch.autograd.grad(loss, (*leaves, tx, rx))

        topology = differentiable.evaluated.topology
        geometry = differentiable.evaluated.geometry
        fields = differentiable.evaluated.fields
        with torch.autograd.forward_ad.dual_level():
            dual_values = tuple(
                torch.autograd.forward_ad.make_dual(value, torch.ones_like(value))
                for value in leaves
            )
            dual_geometry = PathGeometry(
                row_identity=topology.row_identity,
                path_length_m=geometry.path_length_m,
                delay_s=dual_values[0],
                field_direction=dual_values[1],
                interaction_position=geometry.interaction_position,
                interaction_normal=geometry.interaction_normal,
                interaction_positions=dual_values[2],
                interaction_normals=dual_values[3],
            )
            dual_fields = PathFields(
                row_identity=topology.row_identity,
                path_gain=fields.path_gain,
                path_field=fields.path_field,
                field_xyz=dual_values[5],
                coefficient=dual_values[4],
            )
            dual_result = from_capacity_evaluated_paths(
                CapacityEvaluatedPaths(
                    selection=differentiable.selection,
                    evaluated=EvaluatedPaths(
                        topology=topology,
                        geometry=dual_geometry,
                        fields=dual_fields,
                    ),
                ),
                num_rx=2,
                num_tx=2,
                tx_positions=torch.autograd.forward_ad.make_dual(
                    tx, torch.ones_like(tx)
                ),
                rx_positions=torch.autograd.forward_ad.make_dual(
                    rx, torch.ones_like(rx)
                ),
            )
            tangents = tuple(
                torch.autograd.forward_ad.unpack_dual(getattr(dual_result, name)).tangent
                for name in (
                    "a",
                    "tau",
                    "theta_t",
                    "phi_t",
                    "theta_r",
                    "phi_r",
                    "position",
                    "normal",
                    "field_xyz",
                    "field_direction",
                )
            )
        finished = torch.cuda.Event()
        finished.record(stream)
    finished.synchronize()

    _assert_canonical_inert(result, capacity=capacity)
    for gradient in gradients:
        _assert_positive_zero(gradient)
    for tangent in tangents:
        assert tangent is not None
        _assert_positive_zero(tangent)


def test_path_result_constructor_does_not_recompute_device_cardinality() -> None:
    _, packed, _ = _packed()
    result, _, _ = _result(packed)
    mismatched = replace(result, num_paths=torch.zeros_like(result.num_paths))
    assert mismatched.num_paths.shape == result.path_count_shape
    assert mismatched.valid is result.valid


def test_path_result_capacity_zero_capacity_and_nondefault_stream() -> None:
    candidates = _candidate_paths()
    empty_topology = replace(
        candidates.topology,
        valid=candidates.topology.valid[:0],
        tx_id=candidates.topology.tx_id[:0],
        rx_id=candidates.topology.rx_id[:0],
        depth=candidates.topology.depth[:0],
        component_id=candidates.topology.component_id[:0],
        primitive_id=candidates.topology.primitive_id[:0],
        edge_id=candidates.topology.edge_id[:0],
        material_id=candidates.topology.material_id[:0],
        primitive_sequence=candidates.topology.primitive_sequence[:0],
        material_sequence=candidates.topology.material_sequence[:0],
        interaction_type=candidates.topology.interaction_type[:0],
    )
    empty_geometry = PathGeometry(
        row_identity=empty_topology.row_identity,
        path_length_m=candidates.geometry.path_length_m[:0],
        delay_s=candidates.geometry.delay_s[:0],
        field_direction=candidates.geometry.field_direction[:0],
        interaction_position=candidates.geometry.interaction_position[:0],
        interaction_normal=candidates.geometry.interaction_normal[:0],
        interaction_positions=candidates.geometry.interaction_positions[:0],
        interaction_normals=candidates.geometry.interaction_normals[:0],
    )
    empty_fields = PathFields(
        row_identity=empty_topology.row_identity,
        path_gain=candidates.fields.path_gain[:0],
        path_field=candidates.fields.path_field[:0],
        field_xyz=candidates.fields.field_xyz[:0],
        coefficient=candidates.fields.coefficient[:0],
    )
    empty = EvaluatedPaths(
        topology=empty_topology, geometry=empty_geometry, fields=empty_fields
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        state = create_capacity_failure_state(empty.topology.valid)
        packed = evaluated_paths_capacity_pack(
            empty,
            failure_state=state,
            pair_count=4,
            num_tx=2,
            num_rx=2,
            path_capacity_per_pair=0,
        )
        result, _, _ = _result(packed)
        marker = torch.ones((), device="cuda")
    torch.cuda.current_stream().wait_stream(stream)
    assert marker.item() == 1.0
    assert result.a.shape == (2, 1, 2, 1, 0, 1)
    assert result.valid.numel() == 0
    assert result.num_paths.reshape(-1).tolist() == [0, 0, 0, 0]


def test_endpoint_angle_zero_tiny_inf_and_nan_match_torch_branches() -> None:
    _, packed, _ = _packed()
    topology = packed.evaluated.topology
    geometry = packed.evaluated.geometry
    positions = geometry.interaction_positions.detach().clone()
    depth = topology.depth.clone()
    tx = torch.tensor(
        [[-0.0, 0.0, -torch.finfo(torch.float32).tiny], [0.0, 0.0, 0.0]],
        device="cuda",
    )
    rx = torch.tensor(
        [[0.0, -0.0, 0.0], [float("nan"), 0.0, 0.0]], device="cuda"
    )

    # Row 0 is signed-zero/tiny direct geometry. Row 1 has distinct first/last
    # vertices equal to its endpoints, producing true zero directions. Row 3
    # has infinity in both endpoint directions, while row 9 carries NaN.
    depth[0] = 0
    depth[1] = 2
    positions[1, 0] = tx[0]
    positions[1, 1] = rx[0]
    positions[3, 0] = torch.tensor([float("inf"), 0.0, 0.0], device="cuda")
    special_topology = replace(topology, depth=depth)
    special_geometry = PathGeometry(
        row_identity=special_topology.row_identity,
        path_length_m=geometry.path_length_m,
        delay_s=geometry.delay_s,
        field_direction=geometry.field_direction,
        interaction_position=geometry.interaction_position,
        interaction_normal=geometry.interaction_normal,
        interaction_positions=positions,
        interaction_normals=geometry.interaction_normals,
    )
    special_fields = PathFields(
        row_identity=special_topology.row_identity,
        path_gain=packed.evaluated.fields.path_gain,
        path_field=packed.evaluated.fields.path_field,
        field_xyz=packed.evaluated.fields.field_xyz,
        coefficient=packed.evaluated.fields.coefficient,
    )
    special = CapacityEvaluatedPaths(
        selection=packed.selection,
        evaluated=EvaluatedPaths(
            topology=special_topology,
            geometry=special_geometry,
            fields=special_fields,
        ),
    )
    result = from_capacity_evaluated_paths(
        special,
        num_rx=2,
        num_tx=2,
        tx_positions=tx,
        rx_positions=rx,
    )
    departure = torch.stack(
        (
            rx[0] - tx[0],
            positions[1, 0] - tx[0],
            positions[3, 0] - tx[1],
            positions[9, 0] - tx[1],
        )
    )
    receiver = torch.stack(
        (
            tx[0] - rx[0],
            positions[1, 1] - rx[0],
            positions[3, 0] - rx[0],
            positions[9, 0] - rx[1],
        )
    )
    expected_theta_t, expected_phi_t = endpoint_angles(departure)
    expected_theta_r, expected_phi_r = endpoint_angles(receiver)
    rows = torch.tensor([0, 1, 3, 9], device="cuda")
    for name, expected in (
        ("theta_t", expected_theta_t),
        ("phi_t", expected_phi_t),
        ("theta_r", expected_theta_r),
        ("phi_r", expected_phi_r),
    ):
        _assert_float_branch_parity(_flat(result, name)[rows], expected)
    assert torch.count_nonzero(departure[1]).item() == 0
    assert torch.count_nonzero(receiver[1]).item() == 0
    assert torch.isinf(departure[2, 0])
    assert torch.isinf(receiver[2, 0])
    assert torch.isnan(receiver[3, 0])


def _single_direct_capacity() -> CapacityEvaluatedPaths:
    source = _candidate_paths()
    index = slice(2, 3)

    def row(value: torch.Tensor) -> torch.Tensor:
        return value[index].contiguous()

    source_topology = source.topology
    topology = PathTopology(
        valid=row(source_topology.valid),
        tx_id=torch.zeros_like(row(source_topology.tx_id)),
        rx_id=torch.zeros_like(row(source_topology.rx_id)),
        depth=torch.zeros_like(row(source_topology.depth)),
        component_id=row(source_topology.component_id),
        primitive_id=row(source_topology.primitive_id),
        edge_id=row(source_topology.edge_id),
        material_id=row(source_topology.material_id),
        primitive_sequence=row(source_topology.primitive_sequence),
        material_sequence=row(source_topology.material_sequence),
        interaction_type=row(source_topology.interaction_type),
    )
    source_geometry = source.geometry
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=row(source_geometry.path_length_m),
        delay_s=row(source_geometry.delay_s),
        field_direction=row(source_geometry.field_direction),
        interaction_position=row(source_geometry.interaction_position),
        interaction_normal=row(source_geometry.interaction_normal),
        interaction_positions=row(source_geometry.interaction_positions),
        interaction_normals=row(source_geometry.interaction_normals),
    )
    source_fields = source.fields
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=row(source_fields.path_gain),
        path_field=row(source_fields.path_field),
        field_xyz=row(source_fields.field_xyz),
        coefficient=row(source_fields.coefficient),
    )
    state = create_capacity_failure_state(topology.valid)
    return evaluated_paths_capacity_pack(
        EvaluatedPaths(topology=topology, geometry=geometry, fields=fields),
        failure_state=state,
        pair_count=1,
        num_tx=1,
        num_rx=1,
        path_capacity_per_pair=1,
    )


def _angle_outputs(result) -> torch.Tensor:
    return torch.stack(
        (
            result.theta_t.reshape(()),
            result.phi_t.reshape(()),
            result.theta_r.reshape(()),
            result.phi_r.reshape(()),
        )
    )


def _torch_direct_angles(
    tx_positions: torch.Tensor, rx_positions: torch.Tensor
) -> torch.Tensor:
    direction = rx_positions[0] - tx_positions[0]
    theta_t, phi_t = endpoint_angles(direction[None])
    theta_r, phi_r = endpoint_angles((-direction)[None])
    return torch.stack((theta_t[0], phi_t[0], theta_r[0], phi_r[0]))


@pytest.mark.parametrize(
    "direction",
    (
        (0.0, 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
    ),
    ids=("true-zero", "infinity"),
)
def test_endpoint_angle_singular_vjp_and_jvp_match_torch_branches(direction) -> None:
    packed = _single_direct_capacity()
    weights = torch.tensor([1.0, 0.7, 0.9, 1.1], device="cuda")
    tangent_tx = torch.tensor([[0.25, -0.5, 0.75]], device="cuda")
    tangent_rx = torch.tensor([[-0.4, 0.2, 0.6]], device="cuda")

    tx_native = torch.zeros((1, 3), device="cuda", requires_grad=True)
    rx_native = torch.tensor(
        [direction], device="cuda", dtype=torch.float32, requires_grad=True
    )
    native_result = from_capacity_evaluated_paths(
        packed,
        num_rx=1,
        num_tx=1,
        tx_positions=tx_native,
        rx_positions=rx_native,
    )
    native_vjp = torch.autograd.grad(
        (_angle_outputs(native_result) * weights).sum(), (tx_native, rx_native)
    )

    tx_reference = tx_native.detach().clone().requires_grad_()
    rx_reference = rx_native.detach().clone().requires_grad_()
    reference_vjp = torch.autograd.grad(
        (_torch_direct_angles(tx_reference, rx_reference) * weights).sum(),
        (tx_reference, rx_reference),
    )
    for actual, expected in zip(native_vjp, reference_vjp, strict=True):
        _assert_float_branch_parity(actual, expected)

    with torch.autograd.forward_ad.dual_level():
        dual_result = from_capacity_evaluated_paths(
            packed,
            num_rx=1,
            num_tx=1,
            tx_positions=torch.autograd.forward_ad.make_dual(
                tx_native.detach(), tangent_tx
            ),
            rx_positions=torch.autograd.forward_ad.make_dual(
                rx_native.detach(), tangent_rx
            ),
        )
        native_jvp = torch.autograd.forward_ad.unpack_dual(
            _angle_outputs(dual_result)
        ).tangent
    with torch.autograd.forward_ad.dual_level():
        reference_jvp = torch.autograd.forward_ad.unpack_dual(
            _torch_direct_angles(
                torch.autograd.forward_ad.make_dual(
                    tx_reference.detach(), tangent_tx
                ),
                torch.autograd.forward_ad.make_dual(
                    rx_reference.detach(), tangent_rx
                ),
            )
        ).tangent
    assert native_jvp is not None
    assert reference_jvp is not None
    _assert_float_branch_parity(native_jvp, reference_jvp)


def _clone_continuous_capacity(
    packed: CapacityEvaluatedPaths,
) -> tuple[CapacityEvaluatedPaths, tuple[torch.Tensor, ...]]:
    topology = packed.evaluated.topology
    geometry = packed.evaluated.geometry
    fields = packed.evaluated.fields
    leaves = tuple(
        value.detach().clone().requires_grad_()
        for value in (
            geometry.delay_s,
            geometry.field_direction,
            geometry.interaction_positions,
            geometry.interaction_normals,
            fields.coefficient,
            fields.field_xyz,
        )
    )
    cloned_geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=geometry.path_length_m,
        delay_s=leaves[0],
        field_direction=leaves[1],
        interaction_position=geometry.interaction_position,
        interaction_normal=geometry.interaction_normal,
        interaction_positions=leaves[2],
        interaction_normals=leaves[3],
    )
    cloned_fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=fields.path_gain,
        path_field=fields.path_field,
        field_xyz=leaves[5],
        coefficient=leaves[4],
    )
    return (
        CapacityEvaluatedPaths(
            selection=packed.selection,
            evaluated=EvaluatedPaths(
                topology=topology, geometry=cloned_geometry, fields=cloned_fields
            ),
        ),
        leaves,
    )


def _loss(result) -> torch.Tensor:
    valid = result.valid
    return (
        result.a[valid].real.sum()
        + 0.5 * result.a[valid].imag.sum()
        + result.tau[valid].sum()
        + result.theta_t[valid].sum()
        + 0.7 * result.phi_t[valid].sum()
        + 0.9 * result.theta_r[valid].sum()
        + 1.1 * result.phi_r[valid].sum()
        + 0.03 * result.position[valid].sum()
        + 0.04 * result.normal[valid].sum()
        + 0.05 * result.field_xyz[valid].real.sum()
        + 0.06 * result.field_xyz[valid].imag.sum()
        + 0.07 * result.field_direction[valid].sum()
    )


def test_path_result_capacity_vjp_matches_torch_reference() -> None:
    _, packed, _ = _packed()
    native_packed, native_leaves = _clone_continuous_capacity(packed)
    tx_native, rx_native = _endpoints(differentiable=True)
    native_result = from_capacity_evaluated_paths(
        native_packed,
        num_rx=2,
        num_tx=2,
        tx_positions=tx_native,
        rx_positions=rx_native,
    )
    native_grad = torch.autograd.grad(
        _loss(native_result), (*native_leaves, tx_native, rx_native)
    )

    reference_packed, reference_leaves = _clone_continuous_capacity(packed)
    tx_reference, rx_reference = _endpoints(differentiable=True)
    topology = reference_packed.evaluated.topology
    geometry = reference_packed.evaluated.geometry
    fields = reference_packed.evaluated.fields
    valid = topology.valid
    tx = tx_reference[topology.tx_id[valid].to(torch.int64)]
    rx = rx_reference[topology.rx_id[valid].to(torch.int64)]
    direct = rx - tx
    first = geometry.interaction_positions[valid, 0]
    selected = geometry.interaction_positions[valid]
    last_slot = topology.depth[valid].to(torch.int64).clamp(min=1) - 1
    last = selected[torch.arange(selected.shape[0], device="cuda"), last_slot]
    has_interaction = topology.depth[valid] > 0
    departure = torch.where(has_interaction[:, None], first - tx, direct)
    arrival = torch.where(has_interaction[:, None], rx - last, direct)
    theta_t, phi_t = endpoint_angles(departure)
    theta_r, phi_r = endpoint_angles(-arrival)
    clean_normals = torch.where(
        (topology.interaction_type[valid] == int(InteractionType.DIFFRACTION))[
            ..., None
        ]
        & ~torch.isfinite(geometry.interaction_normals[valid]),
        torch.zeros_like(geometry.interaction_normals[valid]),
        geometry.interaction_normals[valid],
    )
    reference_loss = (
        fields.coefficient[valid].real.sum()
        + 0.5 * fields.coefficient[valid].imag.sum()
        + geometry.delay_s[valid].sum()
        + theta_t.sum()
        + 0.7 * phi_t.sum()
        + 0.9 * theta_r.sum()
        + 1.1 * phi_r.sum()
        + 0.03 * geometry.interaction_positions[valid].sum()
        + 0.04 * clean_normals.sum()
        + 0.05 * fields.field_xyz[valid].real.sum()
        + 0.06 * fields.field_xyz[valid].imag.sum()
        + 0.07 * geometry.field_direction[valid].sum()
    )
    reference_grad = torch.autograd.grad(
        reference_loss, (*reference_leaves, tx_reference, rx_reference)
    )
    for actual, expected in zip(native_grad, reference_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-6)
    invalid = ~valid
    for gradient in native_grad[:6]:
        invalid_gradient = gradient[invalid]
        assert torch.isfinite(invalid_gradient).all()
        assert torch.count_nonzero(invalid_gradient).item() == 0


def test_path_result_capacity_endpoint_vjp_is_bitwise_current_torch_reduction() -> None:
    """Stop gate: endpoint reductions must retain current Torch result bits."""

    _, packed, _ = _packed()
    tx_native, rx_native = _endpoints(differentiable=True)
    native_result = from_capacity_evaluated_paths(
        packed,
        num_rx=2,
        num_tx=2,
        tx_positions=tx_native,
        rx_positions=rx_native,
    )
    native_valid = native_result.valid
    native_angle_loss = (
        native_result.theta_t[native_valid].sum()
        + 0.7 * native_result.phi_t[native_valid].sum()
        + 0.9 * native_result.theta_r[native_valid].sum()
        + 1.1 * native_result.phi_r[native_valid].sum()
    )
    native_tx_vjp, native_rx_vjp = torch.autograd.grad(
        native_angle_loss, (tx_native, rx_native)
    )

    tx_reference, rx_reference = _endpoints(differentiable=True)
    topology = packed.evaluated.topology
    geometry = packed.evaluated.geometry
    valid = topology.valid
    tx = tx_reference[topology.tx_id[valid].to(torch.int64)]
    rx = rx_reference[topology.rx_id[valid].to(torch.int64)]
    direct = rx - tx
    selected = geometry.interaction_positions[valid]
    first = selected[:, 0]
    last_slot = topology.depth[valid].to(torch.int64).clamp(min=1) - 1
    last = selected[torch.arange(selected.shape[0], device="cuda"), last_slot]
    has_interaction = topology.depth[valid] > 0
    departure = torch.where(has_interaction[:, None], first - tx, direct)
    receiver_direction = torch.where(
        has_interaction[:, None], last - rx, -direct
    )
    theta_t, phi_t = endpoint_angles(departure)
    theta_r, phi_r = endpoint_angles(receiver_direction)
    reference_angle_loss = (
        theta_t.sum()
        + 0.7 * phi_t.sum()
        + 0.9 * theta_r.sum()
        + 1.1 * phi_r.sum()
    )
    reference_tx_vjp, reference_rx_vjp = torch.autograd.grad(
        reference_angle_loss, (tx_reference, rx_reference)
    )

    for name, actual, expected in (
        ("tx_positions", native_tx_vjp, reference_tx_vjp),
        ("rx_positions", native_rx_vjp, reference_rx_vjp),
    ):
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), (
            f"{name} VJP changed current Torch reduction bits; ADR-029 activation "
            "must stop instead of accepting a tolerance"
        )


def test_path_result_capacity_jvp_vjp_duality() -> None:
    _, packed, _ = _packed()
    differentiable, leaves = _clone_continuous_capacity(packed)
    tx, rx = _endpoints(differentiable=True)
    tangent_leaves = tuple(torch.randn_like(value) for value in leaves)
    tangent_tx = torch.randn_like(tx)
    tangent_rx = torch.randn_like(rx)
    with torch.autograd.forward_ad.dual_level():
        dual_values = tuple(
            torch.autograd.forward_ad.make_dual(primal, tangent)
            for primal, tangent in zip(leaves, tangent_leaves, strict=True)
        )
        topology = differentiable.evaluated.topology
        geometry = differentiable.evaluated.geometry
        fields = differentiable.evaluated.fields
        dual_geometry = PathGeometry(
            row_identity=topology.row_identity,
            path_length_m=geometry.path_length_m,
            delay_s=dual_values[0],
            field_direction=dual_values[1],
            interaction_position=geometry.interaction_position,
            interaction_normal=geometry.interaction_normal,
            interaction_positions=dual_values[2],
            interaction_normals=dual_values[3],
        )
        dual_fields = PathFields(
            row_identity=topology.row_identity,
            path_gain=fields.path_gain,
            path_field=fields.path_field,
            field_xyz=dual_values[5],
            coefficient=dual_values[4],
        )
        dual_packed = CapacityEvaluatedPaths(
            selection=differentiable.selection,
            evaluated=EvaluatedPaths(
                topology=topology, geometry=dual_geometry, fields=dual_fields
            ),
        )
        dual_result = from_capacity_evaluated_paths(
            dual_packed,
            num_rx=2,
            num_tx=2,
            tx_positions=torch.autograd.forward_ad.make_dual(tx, tangent_tx),
            rx_positions=torch.autograd.forward_ad.make_dual(rx, tangent_rx),
        )
        _, tangent_loss = torch.autograd.forward_ad.unpack_dual(_loss(dual_result))
    primal_result = from_capacity_evaluated_paths(
        differentiable,
        num_rx=2,
        num_tx=2,
        tx_positions=tx,
        rx_positions=rx,
    )
    gradients = torch.autograd.grad(_loss(primal_result), (*leaves, tx, rx))
    expected = sum(
        (gradient.conj() * tangent).real.sum()
        for gradient, tangent in zip(
            gradients, (*tangent_leaves, tangent_tx, tangent_rx), strict=True
        )
    )
    torch.testing.assert_close(tangent_loss, expected, rtol=4.0e-5, atol=4.0e-6)


def test_path_result_capacity_angle_gradcheck() -> None:
    candidates = _candidate_paths()
    source_topology = candidates.topology
    index = slice(4, 5)
    topology = PathTopology(
        valid=source_topology.valid[index],
        tx_id=source_topology.tx_id[index],
        rx_id=source_topology.rx_id[index],
        depth=source_topology.depth[index],
        component_id=source_topology.component_id[index],
        primitive_id=source_topology.primitive_id[index],
        edge_id=source_topology.edge_id[index],
        material_id=source_topology.material_id[index],
        primitive_sequence=source_topology.primitive_sequence[index],
        material_sequence=source_topology.material_sequence[index],
        interaction_type=source_topology.interaction_type[index],
    )
    source_geometry = candidates.geometry
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=source_geometry.path_length_m[index],
        delay_s=source_geometry.delay_s[index],
        field_direction=source_geometry.field_direction[index],
        interaction_position=source_geometry.interaction_position[index],
        interaction_normal=source_geometry.interaction_normal[index],
        interaction_positions=source_geometry.interaction_positions[index],
        interaction_normals=source_geometry.interaction_normals[index],
    )
    source_fields = candidates.fields
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=source_fields.path_gain[index],
        path_field=source_fields.path_field[index],
        field_xyz=source_fields.field_xyz[index],
        coefficient=source_fields.coefficient[index],
    )
    state = create_capacity_failure_state(topology.valid)
    packed = evaluated_paths_capacity_pack(
        EvaluatedPaths(topology=topology, geometry=geometry, fields=fields),
        failure_state=state,
        pair_count=1,
        num_tx=1,
        num_rx=1,
        path_capacity_per_pair=1,
    )
    base_positions = (
        packed.evaluated.geometry.interaction_positions.detach()
        .clone()
        .requires_grad_()
    )
    base_tx = torch.tensor(
        [[-2.0, 1.0, 0.5]], device="cuda", requires_grad=True
    )
    base_rx = torch.tensor(
        [[12.0, -3.0, 5.0]], device="cuda", requires_grad=True
    )

    def angles(
        interaction_positions: torch.Tensor,
        tx_positions: torch.Tensor,
        rx_positions: torch.Tensor,
    ) -> torch.Tensor:
        packed_geometry = packed.evaluated.geometry
        differentiable_geometry = PathGeometry(
            row_identity=packed.evaluated.topology.row_identity,
            path_length_m=packed_geometry.path_length_m,
            delay_s=packed_geometry.delay_s,
            field_direction=packed_geometry.field_direction,
            interaction_position=packed_geometry.interaction_position,
            interaction_normal=packed_geometry.interaction_normal,
            interaction_positions=interaction_positions,
            interaction_normals=packed_geometry.interaction_normals,
        )
        differentiable = CapacityEvaluatedPaths(
            selection=packed.selection,
            evaluated=EvaluatedPaths(
                topology=packed.evaluated.topology,
                geometry=differentiable_geometry,
                fields=packed.evaluated.fields,
            ),
        )
        result = from_capacity_evaluated_paths(
            differentiable,
            num_rx=1,
            num_tx=1,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
        )
        return torch.stack(
            (
                result.theta_t.reshape(()),
                result.phi_t.reshape(()),
                result.theta_r.reshape(()),
                result.phi_r.reshape(()),
            )
        )

    assert torch.autograd.gradcheck(
        angles,
        (base_positions, base_tx, base_rx),
        eps=1.0e-3,
        atol=3.0e-3,
        rtol=2.0e-2,
        fast_mode=True,
        check_forward_ad=True,
    )


def test_path_result_capacity_static_contract_has_no_host_compaction_or_trap() -> None:
    root = Path(__file__).resolve().parents[2]
    python_source = (
        root / "src/witwin/channel/path/capacity.py"
    ).read_text(encoding="utf-8")
    native_source = (
        root / "native/channel/kernels/path_result_capacity_pack.cu"
    ).read_text(encoding="utf-8")
    ad_source = (
        root / "native/channel/kernels/path_result_capacity_pack_ad.cu"
    ).read_text(encoding="utf-8")
    for forbidden in (
        ".item(",
        ".cpu(",
        ".numpy(",
        ".tolist(",
        "repeat_interleave",
        "cudaStreamSynchronize",
        "<<<1, 1",
        "trap;",
        "asm(\"trap;\")",
    ):
        assert forbidden not in python_source
        assert forbidden not in native_source
        assert forbidden not in ad_source
    assert "path_result_capacity_pack" in python_source
    result_source = (
        root / "src/witwin/channel/path/result.py"
    ).read_text(encoding="utf-8")
    assert "self.valid.sum" not in result_source
