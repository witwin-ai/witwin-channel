from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from witwin.channel.kernels import scattering as functional


def _table_args(rows: int = 2) -> list[torch.Tensor]:
    return [
        torch.ones(rows, dtype=torch.bool),
        torch.zeros(rows, 3),
        torch.zeros(rows, 3),
        torch.zeros(2, 1, 2, 2),
        torch.zeros(2, 1, 2, 2),
    ]


def _sample_args(rows: int = 2) -> list[torch.Tensor]:
    return [
        torch.ones(rows, dtype=torch.bool),
        torch.zeros(rows, 3),
        torch.zeros(rows, 2),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 2, 2),
        torch.zeros(1, 2, 2, 2),
    ]


def _event_args(rows: int = 2) -> list[torch.Tensor]:
    return [
        torch.zeros(rows),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.complex64),
        torch.zeros(rows, dtype=torch.complex64),
        torch.zeros(rows, dtype=torch.complex64),
        torch.zeros(rows, dtype=torch.complex64),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.int32),
    ]


def _ensemble_args(rows: int = 2, samples: int = 2) -> list[torch.Tensor]:
    return [
        torch.ones(rows, dtype=torch.bool),
        torch.zeros(rows, 3),
        torch.zeros(rows),
        torch.zeros(rows),
        torch.zeros(samples, 3),
        torch.zeros(samples, 3),
        torch.zeros(samples, 3),
        torch.zeros(samples, 3),
        torch.zeros(samples),
        torch.zeros(samples),
        torch.zeros(samples),
        torch.zeros(samples),
        torch.zeros(samples),
        torch.zeros(samples, dtype=torch.int32),
        torch.zeros(samples, 3),
        torch.zeros(rows, 3),
        torch.zeros(rows, dtype=torch.int64),
        torch.zeros(rows, dtype=torch.int64),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, 4, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    ]


def _patch_args(rows: int = 2) -> list[torch.Tensor]:
    return [
        torch.ones(rows, dtype=torch.bool),
        torch.zeros(rows, 3, 3),
        torch.zeros(rows, 3, 2),
        torch.arange(rows, dtype=torch.int64),
        torch.zeros(rows, 3),
        torch.zeros(rows, 3),
        torch.zeros(rows, 3),
        torch.zeros(rows, dtype=torch.complex64),
        torch.zeros(rows, dtype=torch.complex64),
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(rows),
        torch.zeros(rows),
        torch.zeros(rows, 3),
        torch.zeros(2, 2),
    ]


@pytest.fixture(autouse=True)
def _isolate_thin_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(functional, "validate_cuda_tensor", lambda *_args, **_kwargs: None)
    node = torch.zeros(1)
    monkeypatch.setattr(functional, "_duffy_nodes", lambda _device: (node, node, node))


def _reject_native_call(_name: str) -> Callable[..., object]:
    def unexpected_call(*_args: object) -> object:
        raise AssertionError("native operation must not run after invalid input")

    return unexpected_call


@pytest.mark.parametrize(
    "operation",
    (
        "scattering_table_eval",
        "scattering_table_eval_backward",
        "scattering_table_eval_jvp",
        "scattering_table_sample",
        "scattering_event_probabilities",
        "scattering_ensemble_eval",
        "scattering_ensemble_eval_backward",
        "scattering_ensemble_eval_jvp",
        "scattering_patch_integral_eval",
        "scattering_patch_integral_eval_backward",
        "scattering_patch_integral_eval_jvp",
    ),
)
def test_facades_reject_invalid_native_result_fields(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        functional,
        "_required_native_op",
        lambda name: lambda *_args: {"unexpected": torch.tensor(name == operation)},
    )

    if operation.startswith("scattering_table_eval"):
        args = _table_args()
        kwargs = {}
    elif operation == "scattering_table_sample":
        args = _sample_args()
        kwargs = {}
    elif operation == "scattering_event_probabilities":
        args = _event_args()
        kwargs = {"frequency_hz": 1.0, "probability_floor": 0.0}
    elif operation.startswith("scattering_ensemble_eval"):
        args = _ensemble_args()
        kwargs = {"coef": 1.0, "threshold": 0.0}
    else:
        args = _patch_args()
        kwargs = {"k0": 1.0}
        if operation == "scattering_patch_integral_eval_backward":
            kwargs["grad_total"] = torch.zeros((), dtype=torch.complex64)

    with pytest.raises(TypeError, match=rf"{operation} returned invalid fields"):
        getattr(functional, operation)(*args, **kwargs)


def test_table_eval_rejects_direction_shape_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(functional, "_required_native_op", _reject_native_call)
    args = _table_args()
    args[2] = torch.zeros(3, 3)

    with pytest.raises(ValueError, match=r"matching shape \(N, 3\)"):
        functional.scattering_table_eval(*args)


def test_table_eval_rejects_valid_row_mismatch_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(functional, "_required_native_op", _reject_native_call)
    args = _table_args()
    args[0] = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="valid must match wi rows"):
        functional.scattering_table_eval(*args)


def test_table_pdf_rejects_row_mismatch_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(functional, "_required_native_op", _reject_native_call)
    valid, wi, wo, sample_density, _ = _table_args()
    valid = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="valid, wi, and wo rows must match"):
        functional.scattering_table_pdf(valid, wi, wo, sample_density)


def test_table_sample_rejects_uniform_shape_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(functional, "_required_native_op", _reject_native_call)
    args = _sample_args()
    args[2] = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="valid and uniforms must match wi rows"):
        functional.scattering_table_sample(*args)


def test_ensemble_rejects_valid_row_mismatch_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(functional, "_required_native_op", _reject_native_call)
    args = _ensemble_args()
    args[0] = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="valid must match ensemble rows"):
        functional.scattering_ensemble_eval(*args, coef=1.0, threshold=0.0)


def test_patch_rejects_valid_row_mismatch_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(functional, "_required_native_op", _reject_native_call)
    args = _patch_args()
    args[0] = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="valid must match patch rows"):
        functional.scattering_patch_integral_eval(*args, k0=1.0)
