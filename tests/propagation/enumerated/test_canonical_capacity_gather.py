from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel_native.propagation.enumerated.canonical_capacity import (
    evaluated_paths_canonical_capacity_gather,
)
from witwin.channel_native.propagation.models.capacity import (
    CanonicalEvaluatedPaths,
    CanonicalPathSelection,
)
from witwin.channel_native.propagation.models.evaluated import EvaluatedPaths
from witwin.channel_native.propagation.models.fields import PathFields
from witwin.channel_native.propagation.models.geometry import PathGeometry
from witwin.channel_native.propagation.models.topology import PathTopology
from witwin.channel_native.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    create_capacity_failure_state,
    required_symbol,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


_CONTINUOUS = (
    "path_length_m",
    "delay_s",
    "field_direction",
    "interaction_position",
    "interaction_normal",
    "interaction_positions",
    "interaction_normals",
    "path_gain",
    "path_field",
    "field_xyz",
    "coefficient",
)


def _paths(
    valid_values: list[bool],
    *,
    tx_values: list[int] | None = None,
    rx_values: list[int] | None = None,
    differentiable: bool = False,
) -> EvaluatedPaths:
    count = len(valid_values)
    rows = torch.arange(count, device="cuda", dtype=torch.float32)
    valid = torch.tensor(valid_values, device="cuda", dtype=torch.bool)
    tx_id = torch.tensor(
        tx_values if tx_values is not None else [index % 2 for index in range(count)],
        device="cuda",
        dtype=torch.int32,
    )
    rx_id = torch.tensor(
        rx_values
        if rx_values is not None
        else [(index // 2) % 2 for index in range(count)],
        device="cuda",
        dtype=torch.int32,
    )
    poison = torch.iinfo(torch.int32).min
    tx_id = torch.where(valid, tx_id, torch.full_like(tx_id, poison))
    rx_id = torch.where(valid, rx_id, torch.full_like(rx_id, poison))
    ints = torch.arange(count, device="cuda", dtype=torch.int32)
    sequence = torch.stack((ints + 10, ints + 20), dim=1)
    topology = PathTopology(
        valid=valid,
        tx_id=tx_id,
        rx_id=rx_id,
        depth=(ints % 3).contiguous(),
        component_id=(ints % 7).contiguous(),
        primitive_id=(ints + 30).contiguous(),
        edge_id=(ints + 40).contiguous(),
        material_id=(ints + 50).contiguous(),
        primitive_sequence=sequence.contiguous(),
        material_sequence=(sequence + 100).contiguous(),
        interaction_type=(sequence % 5).contiguous(),
    )
    vec = torch.stack((rows + 0.1, rows + 0.2, rows + 0.3), dim=1)
    seq_vec = torch.stack((vec + 10.0, vec + 20.0), dim=1)
    real_values: dict[str, torch.Tensor] = {
        "path_length_m": rows + 1.0,
        "delay_s": rows + 2.0,
        "field_direction": vec + 3.0,
        "interaction_position": vec + 4.0,
        "interaction_normal": vec + 5.0,
        "interaction_positions": seq_vec + 6.0,
        "interaction_normals": seq_vec + 7.0,
        "path_gain": rows + 8.0,
    }
    complex_rows = torch.complex(rows + 9.0, rows + 10.0)
    complex_values: dict[str, torch.Tensor] = {
        "path_field": complex_rows,
        "field_xyz": torch.complex(vec + 11.0, vec + 12.0),
        "coefficient": complex_rows + complex(13.0, 14.0),
    }
    for values in (real_values, complex_values):
        for name, value in values.items():
            poisoned = value.clone()
            poisoned[~valid] = (
                complex(float("nan"), float("inf"))
                if value.is_complex()
                else float("nan")
            )
            values[name] = poisoned.contiguous().requires_grad_(differentiable)
    geometry = PathGeometry(
        row_identity=topology.row_identity,
        **{name: real_values[name] for name in _CONTINUOUS[:7]},
    )
    fields = PathFields(
        row_identity=topology.row_identity,
        path_gain=real_values["path_gain"],
        path_field=complex_values["path_field"],
        field_xyz=complex_values["field_xyz"],
        coefficient=complex_values["coefficient"],
    )
    return EvaluatedPaths(topology=topology, geometry=geometry, fields=fields)


def _continuous(paths: EvaluatedPaths) -> dict[str, torch.Tensor]:
    return {
        **{name: getattr(paths.geometry, name) for name in _CONTINUOUS[:7]},
        **{name: getattr(paths.fields, name) for name in _CONTINUOUS[7:]},
    }


def _with_continuous(
    paths: EvaluatedPaths,
    values: dict[str, torch.Tensor],
) -> EvaluatedPaths:
    topology = paths.topology
    return EvaluatedPaths(
        topology=topology,
        geometry=PathGeometry(
            row_identity=topology.row_identity,
            **{name: values[name] for name in _CONTINUOUS[:7]},
        ),
        fields=PathFields(
            row_identity=topology.row_identity,
            **{name: values[name] for name in _CONTINUOUS[7:]},
        ),
    )


def _selection(
    paths: EvaluatedPaths,
    *,
    indices: list[int],
    valid_values: list[bool],
    num_paths: list[int],
    num_selected: int | None = None,
    failure_state: CapacityFailureState | None = None,
) -> CanonicalPathSelection:
    capacity = paths.row_count
    assert len(indices) == capacity
    assert len(valid_values) == capacity
    state = failure_state or create_capacity_failure_state(paths.topology.valid)
    return CanonicalPathSelection(
        candidate_capacity=capacity,
        pair_count=4,
        num_tx=2,
        num_rx=2,
        failure_state=state,
        selected_row_index=torch.tensor(indices, device="cuda", dtype=torch.int64),
        valid=torch.tensor(valid_values, device="cuda", dtype=torch.bool),
        num_selected=torch.tensor(
            [sum(valid_values) if num_selected is None else num_selected],
            device="cuda",
            dtype=torch.int32,
        ),
        num_paths=torch.tensor(num_paths, device="cuda", dtype=torch.int32),
    )


def _gather(
    paths: EvaluatedPaths,
) -> tuple[CanonicalPathSelection, CanonicalEvaluatedPaths]:
    selection = _selection(
        paths,
        indices=[3, 0, 2, -1, -1],
        valid_values=[True, True, True, False, False],
        num_paths=[1, 1, 0, 1],
    )
    return selection, evaluated_paths_canonical_capacity_gather(
        paths, selection=selection
    )


def test_canonical_gather_preserves_live_bits_and_shared_row_identity() -> None:
    paths = _paths(
        [True, False, True, True, False],
        tx_values=[0, 0, 1, 1, 0],
        rx_values=[0, 0, 0, 1, 0],
    )
    source_selection, gathered = _gather(paths)
    output_selection = gathered.selection
    output = gathered.evaluated

    assert output_selection is not source_selection
    assert output_selection.valid is not source_selection.valid
    assert output_selection.valid is output.topology.valid
    assert output.geometry.row_identity is output.topology.row_identity
    assert output.fields.row_identity is output.topology.row_identity
    assert output_selection.selected_row_index.tolist() == [3, 0, 2, -1, -1]
    assert output_selection.valid.tolist() == [True, True, True, False, False]
    assert output_selection.num_selected.tolist() == [3]
    assert output_selection.num_paths.tolist() == [1, 1, 0, 1]

    selected = output_selection.selected_row_index[output_selection.valid]
    for name in (
        "tx_id",
        "rx_id",
        "depth",
        "component_id",
        "primitive_id",
        "edge_id",
        "material_id",
        "primitive_sequence",
        "material_sequence",
        "interaction_type",
    ):
        assert torch.equal(
            getattr(output.topology, name)[output_selection.valid],
            getattr(paths.topology, name)[selected],
        )
    for name, source in _continuous(paths).items():
        assert torch.equal(
            _continuous(output)[name][output_selection.valid], source[selected]
        )

    invalid = ~output_selection.valid
    for name in (
        "tx_id",
        "rx_id",
        "component_id",
        "primitive_id",
        "edge_id",
        "material_id",
    ):
        assert torch.all(getattr(output.topology, name)[invalid] == -1)
    assert torch.count_nonzero(output.topology.depth[invalid]).item() == 0
    assert torch.all(output.topology.primitive_sequence[invalid] == -1)
    assert torch.all(output.topology.material_sequence[invalid] == -1)
    assert torch.count_nonzero(output.topology.interaction_type[invalid]).item() == 0
    assert torch.all(output.geometry.path_length_m[invalid] == -1)
    assert torch.all(output.geometry.delay_s[invalid] == -1)
    for name in _CONTINUOUS[2:]:
        assert torch.count_nonzero(_continuous(output)[name][invalid]).item() == 0


def test_canonical_gather_empty_capacity_and_nondefault_stream() -> None:
    paths = _paths([])
    selection = CanonicalPathSelection(
        candidate_capacity=0,
        pair_count=4,
        num_tx=2,
        num_rx=2,
        failure_state=create_capacity_failure_state(paths.topology.valid),
        selected_row_index=torch.empty(0, device="cuda", dtype=torch.int64),
        valid=torch.empty(0, device="cuda", dtype=torch.bool),
        num_selected=torch.zeros(1, device="cuda", dtype=torch.int32),
        num_paths=torch.zeros(4, device="cuda", dtype=torch.int32),
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gathered = evaluated_paths_canonical_capacity_gather(
            paths, selection=selection
        )
    stream.synchronize()

    assert gathered.evaluated.row_count == 0
    assert gathered.selection.num_selected.tolist() == [0]
    assert gathered.selection.num_paths.tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "mode",
    (
        "duplicate",
        "non_prefix",
        "bad_index",
        "invalid_source",
        "selected_count",
        "pair_count",
        "bad_endpoint",
    ),
)
def test_canonical_gather_contract_failure_is_fully_inert(mode: str) -> None:
    tx_values = [0, 0, 1, 1]
    if mode == "bad_endpoint":
        tx_values[0] = 9
    paths = _paths(
        [True, False, True, True],
        tx_values=tx_values,
        rx_values=[0, 0, 0, 1],
        differentiable=True,
    )
    indices = [0, 2, -1, -1]
    valid_values = [True, True, False, False]
    num_paths = [1, 1, 0, 0]
    num_selected: int | None = None
    if mode == "duplicate":
        indices[1] = 0
        num_paths = [2, 0, 0, 0]
    elif mode == "non_prefix":
        indices = [0, -1, 2, -1]
        valid_values = [True, False, True, False]
    elif mode == "bad_index":
        indices[1] = 99
    elif mode == "invalid_source":
        indices[1] = 1
        num_paths = [2, 0, 0, 0]
    elif mode == "selected_count":
        num_selected = 1
    elif mode == "pair_count":
        num_paths = [2, 0, 0, 0]
    selection = _selection(
        paths,
        indices=indices,
        valid_values=valid_values,
        num_paths=num_paths,
        num_selected=num_selected,
    )

    gathered = evaluated_paths_canonical_capacity_gather(
        paths, selection=selection
    )

    assert selection.failure_state.bits.tolist() == [
        int(CapacityFailureBit.PAIR_CONTRACT_ERROR)
    ]
    assert gathered.selection.selected_row_index.tolist() == [-1, -1, -1, -1]
    assert gathered.selection.valid.tolist() == [False, False, False, False]
    assert gathered.selection.num_selected.tolist() == [0]
    assert gathered.selection.num_paths.tolist() == [0, 0, 0, 0]
    assert torch.all(gathered.evaluated.topology.tx_id == -1)
    assert torch.all(gathered.evaluated.geometry.path_length_m == -1)
    assert torch.count_nonzero(gathered.evaluated.fields.field_xyz).item() == 0

    gathered.evaluated.geometry.path_length_m.sum().backward()
    for value in _continuous(paths).values():
        assert value.grad is not None
        assert torch.count_nonzero(value.grad).item() == 0


@pytest.mark.parametrize(
    ("mode", "valid_values", "indices"),
    (
        ("duplicate", [True, True, False, False], [0, 0, -1, -1]),
        ("out_of_range", [True, True, False, False], [0, 9, -1, -1]),
        ("non_prefix", [True, False, True, False], [0, -1, 2, -1]),
    ),
)
@pytest.mark.parametrize("companion", ("backward", "jvp"))
def test_canonical_gather_direct_ad_malformed_selection_is_inert(
    mode: str,
    valid_values: list[bool],
    indices: list[int],
    companion: str,
) -> None:
    del mode
    paths = _paths([True, True, True, True])
    state = create_capacity_failure_state(paths.topology.valid)
    values = _continuous(paths)
    raw = required_symbol(
        f"evaluated_paths_canonical_capacity_gather_{companion}"
    )(
        state.bits,
        torch.tensor(valid_values, device="cuda", dtype=torch.bool),
        torch.tensor(indices, device="cuda", dtype=torch.int64),
        *(values[name] for name in _CONTINUOUS),
        4,
        2,
    )

    assert state.bits.tolist() == [int(CapacityFailureBit.PAIR_CONTRACT_ERROR)]
    assert isinstance(raw, dict)
    assert set(raw) == set(_CONTINUOUS)
    for output in raw.values():
        assert torch.count_nonzero(output).item() == 0


def test_canonical_gather_vjp_missing_and_lazy_conjugate_cotangents() -> None:
    paths = _paths(
        [True, False, True, True, False],
        tx_values=[0, 0, 1, 1, 0],
        rx_values=[0, 0, 0, 1, 0],
        differentiable=True,
    )
    _source, gathered = _gather(paths)
    cotangent = torch.tensor(
        [complex(1.0, 2.0), complex(-3.0, 4.0), complex(5.0, -6.0), 0j, 0j],
        device="cuda",
        dtype=torch.complex64,
    ).conj()
    gathered.evaluated.fields.path_field.backward(cotangent)

    selected = gathered.selection.selected_row_index[gathered.selection.valid]
    expected = torch.zeros_like(paths.fields.path_field)
    expected[selected] = cotangent[gathered.selection.valid]
    assert torch.equal(paths.fields.path_field.grad, expected)
    for name, value in _continuous(paths).items():
        if name == "path_field":
            continue
        assert value.grad is not None
        assert torch.count_nonzero(value.grad).item() == 0


def test_canonical_gather_jvp_all_continuous_fields_and_duality() -> None:
    paths = _paths(
        [True, False, True, True, False],
        tx_values=[0, 0, 1, 1, 0],
        rx_values=[0, 0, 0, 1, 0],
    )
    values = _continuous(paths)
    tangents = {
        name: torch.full_like(value, complex(2.0, -0.5) if value.is_complex() else 2.0)
        for name, value in values.items()
    }
    with torch.autograd.forward_ad.dual_level():
        dual_values = {
            name: torch.autograd.forward_ad.make_dual(values[name], tangents[name])
            for name in _CONTINUOUS
        }
        _source, gathered = _gather(_with_continuous(paths, dual_values))
        selected = gathered.selection.selected_row_index
        valid = gathered.selection.valid
        for name, output in _continuous(gathered.evaluated).items():
            _primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
            assert tangent is not None
            assert torch.equal(tangent[valid], tangents[name][selected[valid]])
            assert torch.count_nonzero(tangent[~valid]).item() == 0

    reverse_values = _continuous(paths)
    reverse_values["path_length_m"] = (
        reverse_values["path_length_m"].detach().clone().requires_grad_(True)
    )
    _source, reverse = _gather(_with_continuous(paths, reverse_values))
    weights = torch.arange(5, device="cuda", dtype=torch.float32) + 0.25
    reverse_grad = torch.autograd.grad(
        (reverse.evaluated.geometry.path_length_m * weights).sum(),
        reverse_values["path_length_m"],
    )[0]
    source_tangent = torch.arange(1, 6, device="cuda", dtype=torch.float32)
    rhs = (reverse_grad * source_tangent).sum()
    lhs = (
        source_tangent[reverse.selection.selected_row_index[reverse.selection.valid]]
        * weights[reverse.selection.valid]
    ).sum()
    torch.testing.assert_close(lhs, rhs)


def test_canonical_gather_has_no_host_cardinality_or_intermediate_trap() -> None:
    root = Path(__file__).resolve().parents[3]
    primal = (
        root
        / "native/channel_native/kernels/evaluated_paths_canonical_capacity_gather.cu"
    ).read_text(encoding="utf-8")
    ad = (
        root
        / (
            "native/channel_native/kernels/"
            "evaluated_paths_canonical_capacity_gather_ad.cu"
        )
    ).read_text(encoding="utf-8")
    assert "seen_source" in primal
    assert "seen_source" in ad
    assert "atomicOr(seen_source" in primal
    assert "atomicOr(seen_source" in ad
    assert primal.index("canonical_gather_init_kernel<<<") < primal.index(
        "canonical_selection_structure_kernel<<<"
    )
    assert primal.index("canonical_selection_pair_count_kernel<<<") < primal.index(
        "canonical_gather_publish_kernel<<<"
    )
    for source in (primal, ad):
        assert "failure_state[0] != 0" in source
        assert "trap;" not in source
        for forbidden in ("cudaMemcpy", "cudaStreamSynchronize", ".item", ".cpu"):
            assert forbidden not in source
