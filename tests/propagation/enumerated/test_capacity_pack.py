from __future__ import annotations

from pathlib import Path

import pytest
import torch

from witwin.channel.propagation import enumerated as capacity_module
from witwin.channel.propagation.enumerated import (
    enumerated_capacity_failure_sanitize,
    enumerated_capacity_failure_vector_sanitize,
)
from witwin.channel.propagation.rows import (
    EvaluatedPaths,
    PathFields,
    PathGeometry,
    PathTopology,
)
from witwin.channel.runtime import CapacityFailureBit, create_capacity_failure_state


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


def test_enumerated_failure_sanitizer_vjp_jvp_are_native_and_dual() -> None:
    """The live sanitizer owns evaluated_paths_capacity_pack_{backward,jvp}."""

    reverse = _paths([True, False, True], differentiable=True)
    reverse_state = create_capacity_failure_state(reverse.topology.valid)
    reverse_output = enumerated_capacity_failure_sanitize(
        reverse, failure_state=reverse_state
    )
    weight = torch.arange(
        reverse_output.geometry.path_length_m.numel(),
        device="cuda",
        dtype=torch.float32,
    ) + 0.25
    (reverse_output.geometry.path_length_m * weight).sum().backward()
    gradient = reverse.geometry.path_length_m.grad
    assert gradient is not None
    valid = reverse.topology.valid
    torch.testing.assert_close(gradient[valid], weight[valid])
    assert torch.count_nonzero(gradient[~valid]).item() == 0

    forward = _paths([True, False, True])
    values = _continuous(forward)
    tangent = torch.arange(3, device="cuda", dtype=torch.float32) + 1.0
    with torch.autograd.forward_ad.dual_level():
        dual_values = dict(values)
        dual_values["path_length_m"] = torch.autograd.forward_ad.make_dual(
            values["path_length_m"], tangent
        )
        forward_state = create_capacity_failure_state(forward.topology.valid)
        forward_output = enumerated_capacity_failure_sanitize(
            _with_continuous(forward, dual_values), failure_state=forward_state
        )
        _primal, jvp = torch.autograd.forward_ad.unpack_dual(
            forward_output.geometry.path_length_m
        )
        assert jvp is not None
        torch.testing.assert_close(jvp[valid], tangent[valid])
        assert torch.count_nonzero(jvp[~valid]).item() == 0
        lhs = (jvp * weight).sum()
    rhs = (gradient * tangent).sum()
    torch.testing.assert_close(lhs, rhs)


def test_enumerated_failure_sanitizer_ad_owns_the_native_companions() -> None:
    import inspect

    backward = inspect.getsource(
        capacity_module._EnumeratedCapacityFailureSanitizeFunction.backward
    )
    jvp = inspect.getsource(
        capacity_module._EnumeratedCapacityFailureSanitizeFunction.jvp
    )
    assert "_evaluated_paths_capacity_pack_backward_native" in backward
    assert "_evaluated_paths_capacity_pack_jvp_native" in jvp


def test_enumerated_failure_sanitizer_family_has_no_host_transfer() -> None:
    root = Path(__file__).resolve().parents[3]
    sanitizer = (
        root / "native/channel/kernels/capacity_failure.cu"
    ).read_text(encoding="utf-8")
    sanitizer = sanitizer.split(
        "// ---- Consolidated from enumerated_capacity_failure_sanitize.cu ----", 1
    )[1].split(
        "// ---- Consolidated from mc_capacity_failure_component_maps_sanitize.cu ----", 1
    )[0]
    ad = (
        root / "native/channel/kernels/evaluated_paths.cu"
    ).read_text(encoding="utf-8")
    ad = ad.split(
        "// ---- Consolidated from evaluated_paths_capacity_pack_ad.cu ----", 1
    )[1]
    assert "failure_state[0] != 0" in sanitizer
    assert "if (!valid[destination])" in ad
    for source in (sanitizer, ad):
        for forbidden in ("trap;", "cudaMemcpy", "cudaStreamSynchronize", ".item", ".cpu"):
            assert forbidden not in source
