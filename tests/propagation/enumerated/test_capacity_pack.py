from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel.propagation.enumerated.capacity import (
    enumerated_capacity_failure_sanitize,
    enumerated_capacity_failure_vector_sanitize,
    evaluated_paths_capacity_pack,
)
from witwin.channel.propagation.models.evaluated import EvaluatedPaths
from witwin.channel.propagation.models.fields import PathFields
from witwin.channel.propagation.models.geometry import PathGeometry
from witwin.channel.propagation.models.topology import PathTopology
from witwin.channel.runtime import (
    CapacityFailureBit,
    CapacityFailureState,
    create_capacity_failure_state,
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
                complex(float("nan"), float("nan"))
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
    paths: EvaluatedPaths, values: dict[str, torch.Tensor]
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


def _pack(
    paths: EvaluatedPaths,
    *,
    capacity: int = 3,
    failure_state: CapacityFailureState | None = None,
):
    failure_state = failure_state or create_capacity_failure_state(paths.topology.valid)
    return evaluated_paths_capacity_pack(
        paths,
        failure_state=failure_state,
        pair_count=4,
        num_tx=2,
        num_rx=2,
        path_capacity_per_pair=capacity,
    )


def test_failure_sanitizers_preserve_success_and_zero_invalid_poison() -> None:
    paths = _paths([True, False, True])
    state = create_capacity_failure_state(paths.topology.valid)
    vector = torch.complex(
        torch.arange(18, device="cuda", dtype=torch.float32).reshape(2, 3, 3),
        torch.ones((2, 3, 3), device="cuda"),
    ).transpose(0, 1)

    output = enumerated_capacity_failure_sanitize(paths, failure_state=state)
    vector_output = enumerated_capacity_failure_vector_sanitize(
        vector, failure_state=state
    )

    assert output.topology.valid.tolist() == [True, False, True]
    for name, source in _continuous(paths).items():
        target = _continuous(output)[name]
        torch.testing.assert_close(target[output.topology.valid], source[[0, 2]])
        invalid = target[~output.topology.valid]
        if name in {"path_length_m", "delay_s"}:
            assert torch.equal(invalid, torch.full_like(invalid, -1.0))
        else:
            assert torch.count_nonzero(invalid).item() == 0
    torch.testing.assert_close(vector_output, vector)
    assert vector_output.is_contiguous()


def test_mixed_capacity_failure_makes_paths_and_vector_sidecar_fully_inert() -> None:
    paths = _paths([True, False, True])
    state = create_capacity_failure_state(paths.topology.valid)
    state.bits.fill_(int(CapacityFailureBit.SEGMENT_PENETRATION_FAILURE))
    vector = torch.full((2, 4, 3), complex(float("nan"), float("nan")), device="cuda")

    output = enumerated_capacity_failure_sanitize(paths, failure_state=state)
    vector_output = enumerated_capacity_failure_vector_sanitize(
        vector, failure_state=state
    )

    assert not output.topology.valid.any().item()
    assert output.topology.tx_id.tolist() == [-1, -1, -1]
    assert output.topology.rx_id.tolist() == [-1, -1, -1]
    for name, value in _continuous(output).items():
        if name in {"path_length_m", "delay_s"}:
            assert torch.equal(value, torch.full_like(value, -1.0))
        else:
            assert torch.count_nonzero(value).item() == 0
    assert torch.count_nonzero(vector_output).item() == 0
    assert not torch.isnan(vector_output.real).any().item()
    assert not torch.signbit(vector_output.real).any().item()
    assert not torch.signbit(vector_output.imag).any().item()


def test_vector_failure_sanitizer_is_native_vjp_jvp_identity_on_success() -> None:
    base = torch.complex(
        torch.arange(18, device="cuda", dtype=torch.float32).reshape(2, 3, 3),
        torch.ones((2, 3, 3), device="cuda"),
    )
    values = base.transpose(0, 1).detach().requires_grad_(True)
    state = create_capacity_failure_state(values)
    output = enumerated_capacity_failure_vector_sanitize(values, failure_state=state)
    cotangent = (
        torch.complex(torch.ones_like(output.real), torch.full_like(output.real, 2.0))
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    )
    (gradient,) = torch.autograd.grad(
        output,
        (values,),
        grad_outputs=(cotangent,),
    )
    torch.testing.assert_close(gradient, cotangent)

    tangent = (
        torch.complex(
            torch.full_like(values.real, 3.0), torch.full_like(values.real, -4.0)
        )
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    )
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(values.detach(), tangent)
        dual_output = enumerated_capacity_failure_vector_sanitize(
            dual, failure_state=state
        )
        actual_tangent = torch.autograd.forward_ad.unpack_dual(dual_output).tangent
        assert actual_tangent is not None
        torch.testing.assert_close(actual_tangent, tangent)


def test_capacity_pack_is_exact_pair_major_inert_and_identity_shared() -> None:
    paths = _paths(
        [True, False, True, True, True, False, True],
        tx_values=[1, 0, 0, 1, 0, 0, 1],
        rx_values=[1, 0, 0, 1, 0, 0, 0],
    )

    packed = _pack(paths)
    selection = packed.selection
    output = packed.evaluated

    assert selection.selected_row_index.tolist() == [
        2,
        4,
        -1,
        6,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        3,
        -1,
    ]
    assert selection.num_paths.tolist() == [2, 1, 0, 2]
    assert selection.overflow.tolist() == [False]
    assert selection.valid is output.topology.valid
    assert output.geometry.row_identity is output.topology.row_identity
    assert output.fields.row_identity is output.topology.row_identity
    assert len(set(selection.selected_row_index[selection.valid].tolist())) == 5

    selected = selection.selected_row_index[selection.valid]
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
        torch.testing.assert_close(
            getattr(output.topology, name)[selection.valid],
            getattr(paths.topology, name)[selected],
        )
    for name, source in _continuous(paths).items():
        target = _continuous(output)[name]
        torch.testing.assert_close(target[selection.valid], source[selected])

    invalid = ~selection.valid
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


def test_capacity_pack_handles_zero_capacity_and_current_stream() -> None:
    paths = _paths([False, False])
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packed = _pack(paths, capacity=0)
    stream.synchronize()

    assert packed.selection.selected_row_index.shape == (0,)
    assert packed.selection.valid.shape == (0,)
    assert packed.selection.num_paths.tolist() == [0, 0, 0, 0]
    assert packed.selection.overflow.tolist() == [False]
    assert packed.evaluated.row_count == 0


def test_capacity_pack_backward_none_cotangents_and_invalid_sources_are_zero() -> None:
    paths = _paths(
        [True, False, True, True],
        tx_values=[0, 0, 1, 1],
        rx_values=[0, 0, 0, 1],
        differentiable=True,
    )
    packed = _pack(paths)
    loss = (
        packed.evaluated.geometry.path_length_m.sum()
        + packed.evaluated.fields.path_field.real.sum()
    )
    loss.backward()

    selected = set(packed.selection.selected_row_index[packed.selection.valid].tolist())
    for name, source in _continuous(paths).items():
        assert source.grad is not None
        if name in {"path_length_m", "path_field"}:
            for row in range(source.shape[0]):
                value = source.grad[row]
                if row in selected:
                    torch.testing.assert_close(value, torch.ones_like(value))
                else:
                    assert torch.count_nonzero(value).item() == 0
        else:
            assert torch.count_nonzero(source.grad).item() == 0


def test_capacity_pack_jvp_gathers_every_continuous_field_and_preserves_zero_slots() -> (
    None
):
    paths = _paths(
        [True, False, True, True],
        tx_values=[0, 0, 1, 1],
        rx_values=[0, 0, 0, 1],
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
        packed = _pack(_with_continuous(paths, dual_values))
        selected = packed.selection.selected_row_index
        valid = packed.selection.valid
        for name, output in _continuous(packed.evaluated).items():
            _primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
            assert tangent is not None
            torch.testing.assert_close(tangent[valid], tangents[name][selected[valid]])
            assert torch.count_nonzero(tangent[~valid]).item() == 0


def test_capacity_pack_jvp_vjp_duality_for_real_geometry() -> None:
    base = _paths(
        [True, False, True, True],
        tx_values=[0, 0, 1, 1],
        rx_values=[0, 0, 0, 1],
    )
    values = _continuous(base)
    tangent = torch.arange(1, 5, device="cuda", dtype=torch.float32)
    with torch.autograd.forward_ad.dual_level():
        dual_values = dict(values)
        dual_values["path_length_m"] = torch.autograd.forward_ad.make_dual(
            values["path_length_m"], tangent
        )
        dual_output = _pack(_with_continuous(base, dual_values))
        _primal, jvp = torch.autograd.forward_ad.unpack_dual(
            dual_output.evaluated.geometry.path_length_m
        )
        assert jvp is not None
        weight = torch.arange(jvp.numel(), device="cuda", dtype=torch.float32) + 0.25
        lhs = (jvp * weight).sum()

    reverse_values = _continuous(base)
    reverse_values["path_length_m"] = (
        reverse_values["path_length_m"].detach().clone().requires_grad_(True)
    )
    reverse_output = _pack(_with_continuous(base, reverse_values))
    grad = torch.autograd.grad(
        (reverse_output.evaluated.geometry.path_length_m * weight).sum(),
        reverse_values["path_length_m"],
    )[0]
    rhs = (grad * tangent).sum()
    torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize(
    ("pair_count", "num_tx", "num_rx", "capacity"),
    ((-1, 1, 1, 1), (1, 1, 1, -1), (2, 1, 1, 1)),
)
def test_capacity_pack_rejects_bad_host_metadata_before_native_allocation(
    pair_count: int, num_tx: int, num_rx: int, capacity: int
) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        paths = _paths([])
        evaluated_paths_capacity_pack(
            paths,
            failure_state=create_capacity_failure_state(paths.topology.valid),
            pair_count=pair_count,
            num_tx=num_tx,
            num_rx=num_rx,
            path_capacity_per_pair=capacity,
        )


@pytest.mark.parametrize(
    ("mode", "bit"),
    (
        ("overflow", CapacityFailureBit.PAIR_CAPACITY_OVERFLOW),
        ("bad_id", CapacityFailureBit.PAIR_CONTRACT_ERROR),
    ),
)
def test_capacity_pack_failure_sets_state_and_returns_fully_inert(
    mode: str, bit: CapacityFailureBit
) -> None:
    paths = _paths(
        [True, True, True] if mode == "overflow" else [True, False],
        tx_values=[0, 0, 0] if mode == "overflow" else [3, 0],
        rx_values=[0, 0, 0] if mode == "overflow" else [0, 0],
    )
    failure_state = create_capacity_failure_state(paths.topology.valid)

    packed = evaluated_paths_capacity_pack(
        paths,
        failure_state=failure_state,
        pair_count=1,
        num_tx=1,
        num_rx=1,
        path_capacity_per_pair=2,
    )

    assert packed.selection.layout.failure_state is failure_state
    assert failure_state.bits.tolist() == [int(bit)]
    assert packed.selection.selected_row_index.tolist() == [-1, -1]
    assert packed.selection.valid.tolist() == [False, False]
    assert packed.selection.num_paths.tolist() == [0]
    assert packed.selection.overflow.tolist() == [mode == "overflow"]
    assert torch.all(packed.evaluated.topology.tx_id == -1)
    assert torch.all(packed.evaluated.geometry.path_length_m == -1)
    assert torch.count_nonzero(packed.evaluated.fields.field_xyz).item() == 0


def test_capacity_pack_failure_keeps_vjp_and_jvp_zero_without_state_ad_abi() -> None:
    reverse = _paths([True], tx_values=[3], rx_values=[0], differentiable=True)
    reverse_state = create_capacity_failure_state(reverse.topology.valid)
    reverse_output = evaluated_paths_capacity_pack(
        reverse,
        failure_state=reverse_state,
        pair_count=1,
        num_tx=1,
        num_rx=1,
        path_capacity_per_pair=1,
    )
    reverse_output.evaluated.geometry.path_length_m.sum().backward()
    assert reverse.geometry.path_length_m.grad is not None
    assert torch.count_nonzero(reverse.geometry.path_length_m.grad).item() == 0

    forward = _paths([True], tx_values=[3], rx_values=[0])
    values = _continuous(forward)
    with torch.autograd.forward_ad.dual_level():
        dual_values = dict(values)
        dual_values["path_length_m"] = torch.autograd.forward_ad.make_dual(
            values["path_length_m"], torch.ones_like(values["path_length_m"])
        )
        forward_state = create_capacity_failure_state(forward.topology.valid)
        forward_output = evaluated_paths_capacity_pack(
            _with_continuous(forward, dual_values),
            failure_state=forward_state,
            pair_count=1,
            num_tx=1,
            num_rx=1,
            path_capacity_per_pair=1,
        )
        _primal, tangent = torch.autograd.forward_ad.unpack_dual(
            forward_output.evaluated.geometry.path_length_m
        )
        assert tangent is not None
        assert torch.count_nonzero(tangent).item() == 0


def test_capacity_pack_family_uses_shared_no_trap_helper_without_host_transfer() -> (
    None
):
    root = Path(__file__).resolve().parents[3]
    finalizer = (
        root / "native/channel/kernels/deterministic_capacity_finalize.cu"
    ).read_text(encoding="utf-8")
    packer = (
        root / "native/channel/kernels/evaluated_paths_capacity_pack.cu"
    ).read_text(encoding="utf-8")
    ad = (
        root / "native/channel/kernels/evaluated_paths_capacity_pack_ad.cu"
    ).read_text(encoding="utf-8")
    assert "deterministic_capacity_finalize_no_trap" in finalizer
    assert "deterministic_capacity_finalize_no_trap" in packer
    assert packer.index("evaluated_paths_capacity_init_kernel<<<") < packer.index(
        "deterministic_capacity_finalize_no_trap"
    )
    assert "trap;" not in finalizer
    assert "trap;" not in packer
    assert "failure_state[0] != 0" in packer
    assert "if (!output.valid[destination])" in packer
    assert "if (!valid[destination])" in ad
    for source in (finalizer, packer, ad):
        for forbidden in ("cudaMemcpy", "cudaStreamSynchronize", ".item", ".cpu"):
            assert forbidden not in source
