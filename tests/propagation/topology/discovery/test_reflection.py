# Copyright Xingyu Chen.
# Tests reflection.

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from witwin.channel.interactions import reflection


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
        reflection.topology_kernels,
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


def test_large_trace_batches_every_transmitter_into_one_trace_and_candidate_space(monkeypatch):
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
        reflection.topology_kernels,
        "deterministic_reflection_epc_input_batch",
        build,
    )
    traced = []

    def trace(tx, **kwargs):
        traced.append(tx)
        # Transmitter-major chains: tx 0 reaches groups {1, 2}, tx 1 reaches {0, 1}.
        return torch.tensor([[-1], [2], [1], [2], [1], [1], [-1], [0]])

    requests = list(
        reflection.iter_reflection_order1_epc_requests(
            plan, trace_group_chains=trace, **_inputs()
        )
    )
    assert len(traced) == 1 and traced[0].shape == (2, 3)
    assert [request.depth for request in requests] == [1] * 5
    assert all(request.epc_inputs is sentinel for request in requests)
    # Four (tx, group) sequences times five receivers, four candidates per launch.
    assert [(call["candidate_start"], call["candidate_end"]) for call in calls] == [
        (0, 4),
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
    ]
    for call in calls:
        assert torch.equal(call["sequences"], torch.tensor([[11], [12], [10], [11]]))
        assert call["sequence_tx"].tolist() == [0, 0, 1, 1]
        assert call["tx_positions"].shape == (2, 3)


def test_unique_tx_chains_matches_per_transmitter_unique():
    chains = torch.tensor(
        [
            [[2, 0], [1, 2], [1, 2], [-1, -1], [0, -1]],
            [[1, 1], [2, -1], [1, 1], [0, 2], [-1, -1]],
        ]
    )
    for depth in (1, 2):
        rows = reflection._unique_tx_chains(chains, depth=depth, group_count=3)
        expected = []
        for tx in range(2):
            reached = chains[tx][chains[tx][:, depth - 1] >= 0][:, :depth]
            for row in torch.unique(reached, dim=0).tolist():
                expected.append([tx, *row])
        assert rows.tolist() == expected
    wide = reflection._unique_tx_chains(chains, depth=2, group_count=2**40)
    assert wide.tolist() == reflection._unique_tx_chains(chains, depth=2, group_count=3).tolist()


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
    assert [field.name for field in fields(reflection.ReflectionEpcRequest)] == [
        "depth",
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
    # Pin the reflection owner's exact import set so discovery, geometry, and
    # field orchestration do not acquire unreviewed dependencies.
    assert imports == {
        "__future__",
        "dataclasses",
        "typing",
        "witwin.channel.materials",
        "witwin.channel.kernels",
        "witwin.channel.kernels.topology",
        "witwin.channel.propagation.geometry",
        "witwin.channel.propagation.topology",
        "witwin.channel.scene.endpoints",
    }
    assert not any(module.startswith("witwin.channel.core") for module in imports)


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


def test_exhaustive_order_is_depth_sequence_chunk_candidate_chunk_and_lazy(monkeypatch):
    events = []
    sentinel = {"sentinel": torch.ones(())}
    plan = reflection.ReflectionMultibouncePlan(
        True, 2, torch.tensor([4, 5]), None, 1, 2
    )
    monkeypatch.setattr(reflection, "_MULTIBOUNCE_PAIR_CHUNK_SIZE", 4)
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
        reflection.topology_kernels,
        "deterministic_reflection_epc_input_batch",
        lambda **k: (
            events.append(
                f"build{k['sequences'].shape[0]}:{k['candidate_start']}:{k['candidate_end']}"
                f":{k['sequence_tx'] is None}:{k['tx_positions'].shape[0]}"
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
    # Depth 1: one sequence x 2 tx x 3 rx = 6 candidates in launches of 4.
    assert events[:5] == [
        "count1",
        "record1",
        "chunk1",
        "build1:0:4:True:2",
        "build1:4:6:True:2",
    ]
    # Depth 2: two sequences x 6 endpoint pairs = 12 candidates.
    assert events[5:] == [
        "count2",
        "record2",
        "chunk2",
        "build2:0:4:True:2",
        "build2:4:8:True:2",
        "build2:8:12:True:2",
    ]
    assert [request.depth for request in requests] == [1, 1, 2, 2, 2]


def test_traced_order_is_one_trace_then_depth_with_per_transmitter_sequences(monkeypatch):
    events = []
    plan = reflection.ReflectionMultibouncePlan(
        False, 3, torch.tensor([10, 11, 12]), torch.arange(3), 1, 2
    )
    per_tx = torch.tensor([[2, -1], [1, 2], [1, 2], [-1, -1]])

    def trace(tx, **kwargs):
        events.append(f"trace{tx.shape[0]}")
        return torch.cat((per_tx, per_tx))

    monkeypatch.setattr(
        reflection.topology_kernels,
        "deterministic_reflection_epc_input_batch",
        lambda **k: (
            events.append(f"build{k['sequences'].tolist()}:{k['sequence_tx'].tolist()}")
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
    assert [request.depth for request in requests] == [1, 2]
    assert events == [
        "trace2",
        "record4",
        "build[[11], [12], [11], [12]]:[0, 0, 1, 1]",
        "record2",
        "build[[11, 12], [11, 12]]:[0, 1]",
    ]


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