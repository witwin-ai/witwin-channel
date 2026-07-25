from __future__ import annotations

import pytest
import torch

from witwin.channel.propagation.consumer import _native
from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.models.fields import PathFields
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _paths(
    path_length: torch.Tensor,
    *,
    valid_values: tuple[bool, ...] = (False, True, False, True),
    tx_values: tuple[int, ...] = (0, 1, 0, 0),
    rx_values: tuple[int, ...] = (0, 0, 1, 1),
) -> EvaluatedPaths:
    device = path_length.device
    rows = int(path_length.shape[0])
    valid = torch.tensor(valid_values, device=device)
    topology = PathTopology(
        valid=valid,
        tx_id=torch.tensor(tx_values, device=device, dtype=torch.int32),
        rx_id=torch.tensor(rx_values, device=device, dtype=torch.int32),
        depth=torch.arange(rows, device=device, dtype=torch.int32),
        component_id=torch.zeros(rows, device=device, dtype=torch.int32),
        primitive_id=torch.arange(rows, device=device, dtype=torch.int32),
        edge_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        material_id=torch.full((rows,), -1, device=device, dtype=torch.int32),
        primitive_sequence=torch.full(
            (rows, 1), -1, device=device, dtype=torch.int32
        ),
        material_sequence=torch.full(
            (rows, 1), -1, device=device, dtype=torch.int32
        ),
        interaction_type=torch.zeros(
            (rows, 1), device=device, dtype=torch.int32
        ),
    )
    zeros3 = torch.zeros((rows, 3), device=device)
    zeros13 = torch.zeros((rows, 1, 3), device=device)
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        path_length_m=path_length,
        delay_s=path_length * 0.5,
        field_direction=zeros3,
        interaction_position=zeros3,
        interaction_normal=zeros3,
        interaction_positions=zeros13,
        interaction_normals=zeros13,
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=path_length * 2.0,
        path_field=torch.complex(path_length, path_length),
        field_xyz=torch.complex(zeros3, zeros3),
        coefficient=torch.complex(path_length, -path_length),
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


def test_compact_finalize_radix_order_vjp_and_jvp_use_real_native() -> None:
    source_ids = torch.tensor([101, 102], device="cuda", dtype=torch.int64)
    sink_ids = torch.tensor([201, 202], device="cuda", dtype=torch.int64)
    primal = torch.arange(1, 5, device="cuda", dtype=torch.float32)

    def evaluate(values):
        compact = _native.compact_evaluated_paths(
            _paths(
                values,
                valid_values=(True, True, False, True),
                tx_values=(1, 0, 0, 1),
                rx_values=(1, 0, 1, 1),
            ),
            source_stable_ids=source_ids,
            sink_stable_ids=sink_ids,
        )
        assert compact.path_count == 3
        assert torch.equal(
            compact.pair_index,
            torch.tensor([0, 3, 3], device="cuda", dtype=torch.int64),
        )
        return compact.evaluated.geometry.path_length_m

    values = primal.detach().requires_grad_()
    compact_values = evaluate(values)
    assert torch.equal(
        compact_values,
        torch.tensor([2.0, 1.0, 4.0], device="cuda"),
    )
    compact_values.sum().backward()
    assert torch.equal(values.grad, torch.tensor([1, 1, 0, 1], device="cuda"))

    _, tangent = torch.func.jvp(evaluate, (primal,), (torch.ones_like(primal),))
    assert torch.equal(tangent, torch.ones(3, device="cuda"))


def test_exact_row_mode_aliases_payload_and_has_no_count_boundary() -> None:
    paths = _paths(
        torch.arange(1, 5, device="cuda", dtype=torch.float32),
        valid_values=(True, True, True, True),
        tx_values=(0, 1, 0, 1),
        rx_values=(0, 0, 1, 1),
    )
    compact = _native.compact_evaluated_paths(
        paths,
        source_stable_ids=torch.tensor(
            [101, 102], device="cuda", dtype=torch.int64
        ),
        sink_stable_ids=torch.tensor(
            [201, 202], device="cuda", dtype=torch.int64
        ),
        rows_are_compact=True,
    )

    assert compact.path_count == paths.row_count
    assert compact.evaluated is paths
    assert compact.evaluated.fields.path_field is paths.fields.path_field
    assert compact.count_d2h_copies == 0
    assert compact.count_d2h_bytes == 0
    assert compact.count_synchronizations == 0
    assert compact.native_launch_count == 1
    assert torch.equal(
        compact.pair_index,
        torch.tensor([0, 1, 2, 3], device="cuda"),
    )


def test_compact_finalize_handles_zero_k_without_radix_launch() -> None:
    paths = _paths(
        torch.arange(1, 5, device="cuda", dtype=torch.float32),
        valid_values=(False, False, False, False),
    )

    compact = _native.compact_evaluated_paths(
        paths,
        source_stable_ids=torch.tensor(
            [101, 102], device="cuda", dtype=torch.int64
        ),
        sink_stable_ids=torch.tensor(
            [201, 202], device="cuda", dtype=torch.int64
        ),
    )

    assert compact.path_count == 0
    assert compact.evaluated.row_count == 0
    assert compact.pair_index.shape == (0,)
    assert torch.equal(
        compact.pair_offsets,
        torch.zeros(5, device="cuda", dtype=torch.int64),
    )
    assert compact.count_d2h_copies == 1
    assert compact.count_d2h_bytes == 8
    assert compact.count_synchronizations == 1
