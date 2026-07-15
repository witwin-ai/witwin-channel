from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from witwin.channel_native.propagation.topology.discovery import reflection


def _inputs():
    return {
        "tx_positions": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "rx_positions": torch.arange(15, dtype=torch.float32).reshape(5, 3),
        "tri_a": torch.zeros((3, 3)),
        "normals": torch.ones((3, 3)),
    }


@pytest.mark.parametrize(("count", "exhaustive"), [(4096, True), (4097, False)])
def test_plan_limit_boundary(monkeypatch, count, exhaustive):
    representative = torch.arange(count)
    mapped = representative.reshape(-1, 1)
    monkeypatch.setattr(
        reflection.topology_construction,
        "deterministic_mapped_face_sequence_chunk",
        lambda *args, **kwargs: mapped,
    )
    plan = reflection.prepare_reflection_order1_plan(
        group_count=count,
        representative_faces=representative,
        face_group_id=representative,
    )
    assert plan.exhaustive is exhaustive
    assert (plan.base_sequences is mapped) is exhaustive
    assert (plan.face_group_id is None) is exhaustive


def test_large_trace_unique_minus_one_tx_major_tail_and_identity(monkeypatch):
    representative = torch.tensor([10, 11, 12])
    plan = reflection.ReflectionOrder1Plan(
        exhaustive=False,
        group_count=3,
        representative_faces=representative,
        base_sequences=None,
        face_group_id=torch.arange(3),
    )
    monkeypatch.setattr(reflection, "_MULTIBOUNCE_PAIR_CHUNK_SIZE", 4)
    sentinel: dict[str, torch.Tensor] = {"sentinel": torch.ones(())}
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        reflection.topology_construction,
        "deterministic_reflection_epc_input_batch",
        build,
    )
    traced = []

    def trace(tx, **kwargs):
        traced.append(tx)
        return torch.tensor([[-1], [2], [1], [2]])

    requests = list(
        reflection.iter_reflection_order1_epc_requests(
            plan, trace_group_chains=trace, **_inputs()
        )
    )
    assert [request.tx_index for request in requests] == [0, 0, 0, 1, 1, 1]
    assert all(request.epc_inputs is sentinel for request in requests)
    assert [call["rx_start"] for call in calls] == [0, 2, 4, 0, 2, 4]
    assert [call["rx_end"] for call in calls] == [2, 4, 5, 2, 4, 5]
    assert all(
        torch.equal(call["sequences"], torch.tensor([[11], [12]])) for call in calls
    )
    assert len(traced) == 2


def test_generator_is_lazy_and_callback_errors_propagate():
    plan = reflection.ReflectionOrder1Plan(
        False, 1, torch.tensor([0]), None, torch.tensor([0])
    )
    calls = []

    def trace(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("trace failed")

    iterator = reflection.iter_reflection_order1_epc_requests(
        plan, trace_group_chains=trace, **_inputs()
    )
    assert calls == []
    with pytest.raises(RuntimeError, match="trace failed"):
        next(iterator)


@pytest.mark.parametrize("group_count", [0, -1])
def test_no_group_yields_nothing(group_count):
    plan = reflection.ReflectionOrder1Plan(
        True, group_count, torch.empty(0), torch.empty((0, 1)), None
    )
    assert (
        list(
            reflection.iter_reflection_order1_epc_requests(
                plan, trace_group_chains=lambda *a, **k: None, **_inputs()
            )
        )
        == []
    )


def test_contract_fields_and_forbidden_imports():
    assert is_dataclass(reflection.ReflectionOrder1Plan)
    assert reflection.ReflectionOrder1Plan.__dataclass_params__.frozen
    assert [field.name for field in fields(reflection.ReflectionOrder1Plan)] == [
        "exhaustive",
        "group_count",
        "representative_faces",
        "base_sequences",
        "face_group_id",
    ]
    assert [field.name for field in fields(reflection.ReflectionOrder1EpcRequest)] == [
        "tx_index",
        "tx",
        "epc_inputs",
    ]
    tree = ast.parse(Path(reflection.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        "core" in module or "geometry" in module or "fields" in module
        for module in imports
    )
