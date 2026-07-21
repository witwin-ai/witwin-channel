from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest
import torch

from witwin.channel_native.propagation.geometry.kernels import (
    bridge,
    penetration_autograd,
)
from witwin.channel_native.propagation.models import penetration
from witwin.channel_native.propagation.models.penetration import (
    SegmentPenetrationBackwardResult,
    SegmentPenetrationJvpResult,
    SegmentPenetrationPolicy,
    SegmentPenetrationResult,
    SegmentPenetrationTapeResult,
)
from witwin.channel_native.runtime.capacity import (
    CapacityFailureBit,
    CapacityFailureState,
)


def _failure_state() -> CapacityFailureState:
    state = object.__new__(CapacityFailureState)
    object.__setattr__(state, "bits", torch.zeros(1, dtype=torch.int32))
    return state


def _result_values(rows: int = 2, capacity: int = 3) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((rows, capacity), dtype=torch.bool),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.bool),
        torch.zeros(rows, dtype=torch.bool),
        torch.zeros(rows),
        torch.zeros((rows, 3)),
        torch.full((rows, capacity), -1.0),
        torch.zeros((rows, capacity, 3)),
        torch.zeros((rows, capacity, 3)),
        torch.zeros((rows, capacity, 3)),
        torch.full((rows, capacity), -1, dtype=torch.int32),
    )


def _tape_values(rows: int = 2, capacity: int = 3) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((rows, capacity), -1, dtype=torch.int32),
        torch.zeros((rows, capacity, 2)),
        torch.zeros((rows, capacity)),
        torch.zeros((rows, capacity), dtype=torch.uint8),
        torch.zeros((rows, capacity), dtype=torch.uint8),
        torch.zeros(rows, dtype=torch.bool),
    )


@pytest.fixture
def cpu_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge, "validate_cuda_tensor", lambda *args, **kwargs: args[1])
    monkeypatch.setattr(
        penetration_autograd, "validate_cuda_tensor", lambda *args, **kwargs: args[1]
    )
    monkeypatch.setattr(
        penetration_autograd,
        "_ad_checked_tangent",
        lambda name, tangent, primal_shape: tangent,
    )
    monkeypatch.setattr(
        bridge, "require_capacity_failure_state", lambda state, **kwargs: state
    )
    monkeypatch.setattr(bridge, "_ad_check_optional_grad", lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, "_ad_check_tangent_vec3", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        penetration, "_require_cuda_tensor", lambda name, value, **kwargs: value
    )
    monkeypatch.setattr(
        penetration, "require_capacity_failure_state", lambda state, **kwargs: state
    )


def _install_native(
    monkeypatch: pytest.MonkeyPatch,
    implementations: dict[str, Callable[..., object]],
) -> None:
    def required(name: str) -> Callable[..., object]:
        return implementations[name]

    monkeypatch.setattr(bridge, "_required_native_op", required)


def _request() -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, CapacityFailureState]:
    return (
        object(),
        torch.zeros((2, 3)),
        torch.ones((2, 3)),
        torch.ones(2, dtype=torch.bool),
        _failure_state(),
    )


def test_policy_and_failure_bit_are_stable_and_generation_free() -> None:
    assert list(SegmentPenetrationPolicy) == [
        SegmentPenetrationPolicy.EnumeratedFullDistance,
        SegmentPenetrationPolicy.MonteCarloTargetInset,
    ]
    assert int(SegmentPenetrationPolicy.EnumeratedFullDistance) == 0
    assert int(SegmentPenetrationPolicy.MonteCarloTargetInset) == 1
    assert int(CapacityFailureBit.SEGMENT_PENETRATION_FAILURE) == 1 << 7
    for name in penetration.__all__:
        assert "v2" not in name.casefold()


def test_forward_and_tape_keep_named_order_and_state_identity(
    monkeypatch: pytest.MonkeyPatch, cpu_contracts: None
) -> None:
    scene, origins, targets, active, state = _request()
    result_values = _result_values()
    tape_values = _tape_values()
    calls: list[tuple[str, tuple[object, ...]]] = []

    def primal(*args: object) -> tuple[torch.Tensor, ...]:
        calls.append(("forward", args))
        return result_values

    def tape(*args: object) -> tuple[torch.Tensor, ...]:
        calls.append(("tape", args))
        return (*result_values, *tape_values)

    _install_native(
        monkeypatch,
        {
            "rayd_segment_penetration_forward": primal,
            "rayd_segment_penetration_forward_tape": tape,
        },
    )
    result = bridge.rayd_segment_penetration_forward(
        scene,
        origins,
        targets,
        active,
        input_active_any=True,
        hit_capacity=3,
        policy=SegmentPenetrationPolicy.EnumeratedFullDistance,
        scene_diagonal=4.5,
        failure_state=state,
    )
    taped = bridge.rayd_segment_penetration_forward_tape(
        scene,
        origins,
        targets,
        active,
        input_active_any=True,
        hit_capacity=3,
        policy=SegmentPenetrationPolicy.MonteCarloTargetInset,
        scene_diagonal=4.5,
        failure_state=state,
    )

    assert isinstance(result, SegmentPenetrationResult)
    assert isinstance(taped, SegmentPenetrationTapeResult)
    assert result.failure_state is state
    assert taped.failure_state is state
    assert tuple(getattr(result, name) for name in bridge._SEGMENT_PENETRATION_RESULT_FIELDS) == result_values
    assert tuple(getattr(taped, name) for name in bridge._SEGMENT_PENETRATION_TAPE_FIELDS) == tape_values
    for _, args in calls:
        assert args[1] is origins
        assert args[2] is targets
        assert args[3] is active
        assert args[8] is state.bits
        assert args[9] == 1 << 7
    assert calls[0][1][6] == 0
    assert calls[1][1][6] == 1


def test_backward_and_jvp_flatten_complete_primal_tape_and_return_named_results(
    monkeypatch: pytest.MonkeyPatch, cpu_contracts: None
) -> None:
    scene, origins, targets, active, state = _request()
    result_values = _result_values()
    tape_values = _tape_values()
    taped = SegmentPenetrationTapeResult(
        SegmentPenetrationResult(3, state, *result_values), *tape_values
    )
    grad_vertices = torch.zeros((4, 3))
    grad_origins = torch.zeros_like(origins)
    tangent_values = (
        torch.zeros(2),
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
        torch.zeros((2, 3, 3)),
        torch.zeros((2, 3, 3)),
        torch.zeros((2, 3, 3)),
    )
    calls: dict[str, tuple[object, ...]] = {}

    def backward(*args: object) -> tuple[torch.Tensor | None, ...]:
        calls["backward"] = args
        return grad_vertices, grad_origins, None

    def jvp(*args: object) -> tuple[torch.Tensor, ...]:
        calls["jvp"] = args
        return tangent_values

    _install_native(
        monkeypatch,
        {
            "rayd_segment_penetration_backward": backward,
            "rayd_segment_penetration_jvp": jvp,
        },
    )
    grad_distance = torch.ones(2)
    gradients = bridge.rayd_segment_penetration_backward(
        scene,
        origins,
        targets,
        active,
        input_active_any=True,
        hit_capacity=3,
        policy=SegmentPenetrationPolicy.EnumeratedFullDistance,
        scene_diagonal=4.5,
        failure_state=state,
        tape=taped,
        grad_distance=grad_distance,
        need_grad_vertices=True,
        need_grad_origins=True,
    )
    tangent_origins = torch.ones_like(origins)
    tangents = bridge.rayd_segment_penetration_jvp(
        scene,
        origins,
        targets,
        active,
        input_active_any=True,
        hit_capacity=3,
        policy=SegmentPenetrationPolicy.EnumeratedFullDistance,
        scene_diagonal=4.5,
        failure_state=state,
        tape=taped,
        tangent_origins=tangent_origins,
    )

    assert isinstance(gradients, SegmentPenetrationBackwardResult)
    assert gradients.grad_vertices is grad_vertices
    assert gradients.grad_origins is grad_origins
    assert gradients.grad_targets is None
    assert isinstance(tangents, SegmentPenetrationJvpResult)
    assert tangents.tangent_geometric_normal is tangent_values[-1]
    expected_tape = (*result_values, *tape_values)
    backward_args = calls["backward"]
    jvp_args = calls["jvp"]
    assert backward_args[8] is state.bits
    assert backward_args[10:27] == expected_tape
    assert backward_args[27] is grad_distance
    assert backward_args[33:] == (True, True, False)
    assert jvp_args[8] is state.bits
    assert jvp_args[10:27] == expected_tape
    assert jvp_args[27] is None
    assert jvp_args[28] is tangent_origins
    assert jvp_args[29] is None


def test_companions_reject_a_different_failure_state_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, cpu_contracts: None
) -> None:
    scene, origins, targets, active, state = _request()
    taped = SegmentPenetrationTapeResult(
        SegmentPenetrationResult(3, state, *_result_values()), *_tape_values()
    )
    monkeypatch.setattr(
        bridge,
        "_required_native_op",
        lambda name: pytest.fail(f"unexpected native dispatch: {name}"),
    )
    with pytest.raises(ValueError, match="exact request failure_state"):
        bridge.rayd_segment_penetration_jvp(
            scene,
            origins,
            targets,
            active,
            input_active_any=True,
            hit_capacity=3,
            policy=SegmentPenetrationPolicy.EnumeratedFullDistance,
            scene_diagonal=4.5,
            failure_state=_failure_state(),
            tape=taped,
        )


def test_missing_segment_penetration_family_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch, cpu_contracts: None
) -> None:
    scene, origins, targets, active, state = _request()
    tape = SegmentPenetrationTapeResult(
        SegmentPenetrationResult(3, state, *_result_values()), *_tape_values()
    )

    def missing(name: str) -> object:
        raise RuntimeError(f"missing required native symbol {name}")

    monkeypatch.setattr(bridge, "_required_native_op", missing)
    common = {
        "input_active_any": True,
        "hit_capacity": 3,
        "policy": SegmentPenetrationPolicy.EnumeratedFullDistance,
        "scene_diagonal": 4.5,
        "failure_state": state,
    }
    calls = {
        "rayd_segment_penetration_forward": lambda: (
            bridge.rayd_segment_penetration_forward(
                scene, origins, targets, active, **common
            )
        ),
        "rayd_segment_penetration_forward_tape": lambda: (
            bridge.rayd_segment_penetration_forward_tape(
                scene, origins, targets, active, **common
            )
        ),
        "rayd_segment_penetration_backward": lambda: (
            bridge.rayd_segment_penetration_backward(
                scene, origins, targets, active, tape=tape, **common
            )
        ),
        "rayd_segment_penetration_jvp": lambda: (
            bridge.rayd_segment_penetration_jvp(
                scene, origins, targets, active, tape=tape, **common
            )
        ),
    }
    for symbol, call in calls.items():
        with pytest.raises(RuntimeError, match=symbol):
            call()


def test_request_requires_explicit_policy_and_structural_inactive_mask(
    cpu_contracts: None,
) -> None:
    scene, origins, targets, _active, state = _request()
    with pytest.raises(TypeError, match="SegmentPenetrationPolicy"):
        bridge._segment_penetration_request_args(
            scene,
            origins,
            targets,
            None,
            input_active_any=True,
            hit_capacity=3,
            policy=0,
            scene_diagonal=4.5,
            failure_state=state,
        )
    with pytest.raises(ValueError, match="explicit device input_active mask"):
        bridge._segment_penetration_request_args(
            scene,
            origins,
            targets,
            None,
            input_active_any=False,
            hit_capacity=3,
            policy=SegmentPenetrationPolicy.EnumeratedFullDistance,
            scene_diagonal=4.5,
            failure_state=state,
        )


def test_custom_function_routes_only_to_native_family_facades() -> None:
    source = inspect.getsource(
        penetration_autograd._RaydSegmentPenetrationAdFunction
    )
    assert "rayd_segment_penetration_forward_tape(" in source
    assert "rayd_segment_penetration_backward(" in source
    assert "rayd_segment_penetration_jvp(" in source
    for forbidden in (".item(", ".cpu(", ".numpy(", "finite_difference"):
        assert forbidden not in source


def test_custom_function_executes_native_tape_vjp_and_jvp(
    monkeypatch: pytest.MonkeyPatch, cpu_contracts: None
) -> None:
    scene, origins, targets, active, state = _request()
    vertices = torch.zeros((4, 3), requires_grad=True)
    origins.requires_grad_()
    targets.requires_grad_()
    result_values = _result_values()
    tape_values = _tape_values()
    calls: list[str] = []

    def forward_tape(*args: object) -> tuple[torch.Tensor, ...]:
        calls.append("forward_tape")
        return (*result_values, *tape_values)

    def backward(*args: object) -> tuple[torch.Tensor, ...]:
        calls.append("backward")
        return (
            torch.full_like(vertices, 1.0),
            torch.full_like(origins, 2.0),
            torch.full_like(targets, 3.0),
        )

    jvp_values = (
        torch.full((2,), 4.0),
        torch.full((2, 3), 5.0),
        torch.full((2, 3), 6.0),
        torch.full((2, 3, 3), 7.0),
        torch.full((2, 3, 3), 8.0),
        torch.full((2, 3, 3), 9.0),
    )

    def jvp(*args: object) -> tuple[torch.Tensor, ...]:
        calls.append("jvp")
        return jvp_values

    _install_native(
        monkeypatch,
        {
            "rayd_segment_penetration_forward_tape": forward_tape,
            "rayd_segment_penetration_backward": backward,
            "rayd_segment_penetration_jvp": jvp,
        },
    )
    result = penetration_autograd.rayd_segment_penetration_ad(
        scene,
        vertices,
        origins,
        targets,
        active,
        input_active_any=True,
        hit_capacity=3,
        policy=SegmentPenetrationPolicy.EnumeratedFullDistance,
        scene_diagonal=4.5,
        failure_state=state,
    )
    (
        result.distance.sum()
        + result.direction.sum()
        + result.t.sum()
        + result.position.sum()
        + result.normal.sum()
        + result.geometric_normal.sum()
    ).backward()
    assert torch.equal(vertices.grad, torch.full_like(vertices, 1.0))
    assert torch.equal(origins.grad, torch.full_like(origins, 2.0))
    assert torch.equal(targets.grad, torch.full_like(targets, 3.0))

    primal_origins = origins.detach()
    with torch.autograd.forward_ad.dual_level():
        dual_origins = torch.autograd.forward_ad.make_dual(
            primal_origins, torch.ones_like(primal_origins)
        )
        dual_result = penetration_autograd.rayd_segment_penetration_ad(
            scene,
            vertices.detach(),
            dual_origins,
            targets.detach(),
            active,
            input_active_any=True,
            hit_capacity=3,
            policy=SegmentPenetrationPolicy.EnumeratedFullDistance,
            scene_diagonal=4.5,
            failure_state=state,
        )
        tangent_distance = torch.autograd.forward_ad.unpack_dual(
            dual_result.distance
        ).tangent
        assert torch.equal(tangent_distance, jvp_values[0])
    assert calls == ["forward_tape", "backward", "forward_tape", "jvp"]
