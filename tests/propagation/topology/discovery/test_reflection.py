from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from witwin.channel.propagation.topology.discovery import reflection


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
    assert [field.name for field in fields(reflection.ReflectionMultibouncePlan)] == [
        "exhaustive",
        "group_count",
        "representative_faces",
        "face_group_id",
        "min_depth",
        "max_depth",
    ]
    assert [
        field.name for field in fields(reflection.ReflectionMultibounceEpcRequest)
    ] == ["depth", "tx_index", "tx", "epc_inputs"]
    assert reflection.ReflectionMultibouncePlan.__dataclass_params__.frozen
    assert not hasattr(
        reflection.ReflectionMultibouncePlan(True, 0, torch.empty(0), None, 1, 0),
        "__dict__",
    )
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


@pytest.mark.parametrize(("groups", "exhaustive"), [(316, True), (317, False)])
def test_multibounce_planning_guard_boundary(groups, exhaustive):
    plan = reflection.prepare_reflection_multibounce_plan(
        group_count=groups,
        representative_faces=torch.arange(groups),
        face_group_id=torch.arange(groups),
        min_depth=2,
        max_depth=2,
    )
    assert plan.exhaustive is exhaustive
    assert (plan.face_group_id is None) is exhaustive


def test_multibounce_plan_empty_depth_range_short_circuits_guard(monkeypatch):
    monkeypatch.setattr(
        reflection,
        "_face_sequence_count",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("count")),
    )
    plan = reflection.prepare_reflection_multibounce_plan(
        group_count=1,
        representative_faces=torch.tensor([0]),
        face_group_id=torch.tensor([0]),
        min_depth=3,
        max_depth=2,
    )
    assert plan.exhaustive


def _multibounce_inputs():
    return {
        "tx_positions": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "rx_positions": torch.arange(9, dtype=torch.float32).reshape(3, 3),
        "sequence_reference": torch.ones(1),
        "tri_a": torch.zeros((2, 3)),
        "normals": torch.ones((2, 3)),
    }


def test_exhaustive_order_is_depth_candidate_chunk_rx_tx_and_lazy(monkeypatch):
    events = []
    sentinel = {"sentinel": torch.ones(())}
    plan = reflection.ReflectionMultibouncePlan(
        True, 2, torch.tensor([4, 5]), None, 1, 2
    )
    monkeypatch.setattr(reflection, "_MULTIBOUNCE_PAIR_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        reflection,
        "_face_sequence_count",
        lambda count, depth, **k: events.append(f"count{depth}") or depth,
    )

    def chunks(count, depth, **kwargs):
        events.append(f"chunk{depth}")
        yield torch.arange(depth).reshape(-1, 1)

    monkeypatch.setattr(reflection, "_face_sequence_chunks", chunks)
    monkeypatch.setattr(
        reflection.topology_construction,
        "deterministic_reflection_epc_input_batch",
        lambda **k: (
            events.append(
                f"build{k['sequences'].shape[0]}:{k['rx_start']}:{int(k['tx'][0])}"
            )
            or sentinel
        ),
    )
    iterator = reflection.iter_reflection_multibounce_epc_requests(
        plan,
        trace_group_chains=lambda *a, **k: None,
        record_candidate_count=lambda value: events.append(f"record{value}"),
        **_multibounce_inputs(),
    )
    assert events == []
    requests = list(iterator)
    assert all(request.epc_inputs is sentinel for request in requests)
    assert events[:4] == ["count1", "record1", "chunk1", "build1:0:0"]
    assert events.index("count2") > events.index("build1:2:3")


def test_traced_order_unique_minus_one_and_no_reached_depth(monkeypatch):
    events = []
    plan = reflection.ReflectionMultibouncePlan(
        False, 3, torch.tensor([10, 11, 12]), torch.arange(3), 1, 2
    )
    chains = torch.tensor([[2, -1], [1, 2], [1, 2], [-1, -1]])

    def trace(tx, **kwargs):
        events.append(f"trace{int(tx[0])}")
        return chains

    monkeypatch.setattr(
        reflection.topology_construction,
        "deterministic_reflection_epc_input_batch",
        lambda **k: (
            events.append(f"build{int(k['tx'][0])}:{k['sequences'].tolist()}")
            or {"sentinel": torch.ones(())}
        ),
    )
    requests = list(
        reflection.iter_reflection_multibounce_epc_requests(
            plan,
            trace_group_chains=trace,
            record_candidate_count=lambda value: events.append(f"record{value}"),
            **_multibounce_inputs(),
        )
    )
    assert [request.tx_index for request in requests] == [0, 0, 1, 1]
    assert events[:3] == ["trace0", "record2", "build0:[[11], [12]]"]
    assert "record1" in events
    assert events.index("trace3") > events.index("record1")


def test_traced_no_chain_records_no_candidate_or_request(monkeypatch):
    plan = reflection.ReflectionMultibouncePlan(
        False, 0, torch.empty(0, dtype=torch.long), torch.empty(0), 1, 2
    )
    recorded = []
    assert (
        list(
            reflection.iter_reflection_multibounce_epc_requests(
                plan,
                trace_group_chains=lambda *a, **k: torch.full((2, 2), -1),
                record_candidate_count=recorded.append,
                **_multibounce_inputs(),
            )
        )
        == []
    )
    assert recorded == []


def test_multibounce_callbacks_and_builder_errors_propagate(monkeypatch):
    plan = reflection.ReflectionMultibouncePlan(True, 1, torch.tensor([0]), None, 1, 1)
    monkeypatch.setattr(reflection, "_face_sequence_count", lambda *a, **k: 1)
    with pytest.raises(RuntimeError, match="candidate"):
        next(
            reflection.iter_reflection_multibounce_epc_requests(
                plan,
                trace_group_chains=lambda *a, **k: None,
                record_candidate_count=lambda value: (_ for _ in ()).throw(
                    RuntimeError("candidate")
                ),
                **_multibounce_inputs(),
            )
        )
