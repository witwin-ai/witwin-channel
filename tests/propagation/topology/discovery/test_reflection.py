# Copyright Xingyu Chen.
# Tests reflection candidate-source selection and request planning without the extension.

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from witwin.channel.interactions import reflection


def _groups(count: int = 3) -> reflection.ReflectionPlaneGroups:
    return reflection.ReflectionPlaneGroups(
        group_count=count,
        representative_faces=torch.arange(10, 10 + count),
        face_group_id=torch.arange(count),
        plane_point=torch.zeros((count, 3)),
        plane_normal=torch.ones((count, 3)),
        bounds=torch.zeros((count, 6)),
        occupancy=torch.zeros((count, 32), dtype=torch.int32),
        scene_scale_m=100.0,
    )


def _inputs(rx: int = 5):
    return {
        "tx_positions": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "rx_positions": torch.arange(3 * rx, dtype=torch.float32).reshape(rx, 3),
        "tri_a": torch.zeros((3, 3)),
        "normals": torch.ones((3, 3)),
    }


def _recording_builder(monkeypatch, calls):
    def build(**kwargs):
        calls.append(kwargs)
        return {"sentinel": torch.ones(())}

    monkeypatch.setattr(reflection.topology_kernels, "deterministic_reflection_epc_input_batch", build)


def _fake_tree(monkeypatch, expansions, *, children=None):
    """Expansion that appends every group to every node, as the unpruned tree does."""

    def expand(frontier, groups, *, prune):
        expansions.append(prune)
        rows = []
        for tx, seq in zip(frontier.node_tx.tolist(), frontier.node_sequences.tolist()):
            for g in range(groups.group_count):
                if not seq or seq[-1] != g:
                    rows.append((tx, [*seq, g]))
        if children is not None:
            rows = rows[:children]
        width = int(frontier.node_sequences.shape[1]) + 1
        return reflection._ImageTreeFrontier(
            torch.tensor([r[0] for r in rows], dtype=torch.int32),
            torch.tensor([r[1] for r in rows], dtype=torch.int32).reshape(-1, width),
            torch.zeros((len(rows), 3)),
        )

    monkeypatch.setattr(reflection, "_expand_image_tree", expand)
    monkeypatch.setattr(reflection, "_frontier_node_limit", lambda depth, device: 1 << 30)


def test_discovery_modes_are_validated():
    assert reflection.validate_reflection_discovery("auto") == "auto"
    for bad in ("tree", "", None, "Auto"):
        with pytest.raises(ValueError, match="discovery must be one of"):
            reflection.prepare_reflection_discovery_plan(
                _groups(), min_depth=1, max_depth=1, discovery=bad
            )


def test_record_reports_completeness_per_depth():
    record = reflection.ReflectionDiscoveryRecord({1: "image_tree", 2: "image_tree"})
    assert record.complete
    assert record.metadata("auto") == {
        "requested": "auto",
        "sources": {"1": "image_tree", "2": "image_tree"},
        "complete": True,
    }
    record.sources[3] = "rays"
    assert not record.complete


def test_small_first_depth_pairs_every_group_with_every_endpoint_pair(monkeypatch):
    calls = []
    _recording_builder(monkeypatch, calls)
    _fake_tree(monkeypatch, expansions := [])
    record = reflection.ReflectionDiscoveryRecord()
    plan = reflection.prepare_reflection_discovery_plan(_groups(), min_depth=1, max_depth=1)
    requests = list(
        reflection.iter_reflection_epc_requests(
            plan, trace_group_chains=lambda *a, **k: None, record=record, **_inputs()
        )
    )
    assert [r.depth for r in requests] == [1]
    # No image-tree launch for a depth-1-only solve: the shared plane sequences go straight in.
    assert expansions == []
    (call,) = calls
    assert call["sequence_tx"] is None and call["candidate_sequence"] is None
    assert call["sequences"].tolist() == [[10], [11], [12]]
    assert (call["candidate_start"], call["candidate_end"]) == (0, 2 * 5 * 3)
    assert record.sources == {1: "image_tree"}


def test_auto_depths_come_from_the_pruned_tree_and_do_not_depend_on_max_depth(monkeypatch):
    def run(max_depth):
        calls, expansions = [], []
        _recording_builder(monkeypatch, calls)
        _fake_tree(monkeypatch, expansions)
        record = reflection.ReflectionDiscoveryRecord()
        plan = reflection.prepare_reflection_discovery_plan(
            _groups(), min_depth=2, max_depth=max_depth
        )
        list(
            reflection.iter_reflection_epc_requests(
                plan, trace_group_chains=lambda *a, **k: None, record=record, **_inputs()
            )
        )
        return calls, expansions, record

    calls3, expansions3, record3 = run(3)
    calls2, _expansions2, record2 = run(2)
    assert expansions3 == [True, True, True]
    assert record3.sources == {1: "image_tree", 2: "image_tree", 3: "image_tree"}
    # Depth 2 sees the same per-transmitter sequences whatever the deepest requested depth.
    assert calls2[0]["sequences"].tolist() == calls3[0]["sequences"].tolist()
    assert calls2[0]["sequence_tx"].tolist() == calls3[0]["sequence_tx"].tolist()
    assert calls3[0]["sequence_tx"].tolist() == [0] * 6 + [1] * 6
    assert record2.complete and record3.complete


def test_exhaustive_expands_without_pruning(monkeypatch):
    _recording_builder(monkeypatch, [])
    _fake_tree(monkeypatch, expansions := [])
    record = reflection.ReflectionDiscoveryRecord()
    plan = reflection.prepare_reflection_discovery_plan(
        _groups(), min_depth=2, max_depth=2, discovery="exhaustive"
    )
    list(
        reflection.iter_reflection_epc_requests(
            plan, trace_group_chains=lambda *a, **k: None, record=record, **_inputs()
        )
    )
    assert expansions == [False, False]
    assert record.sources == {1: "exhaustive", 2: "exhaustive"}


def test_exhaustive_over_memory_fails_before_any_request(monkeypatch):
    _recording_builder(monkeypatch, calls := [])
    _fake_tree(monkeypatch, [])
    monkeypatch.setattr(reflection, "_frontier_node_limit", lambda depth, device: 4)
    plan = reflection.prepare_reflection_discovery_plan(
        _groups(), min_depth=2, max_depth=2, discovery="exhaustive"
    )
    with pytest.raises(ValueError, match="discovery='auto'"):
        list(
            reflection.iter_reflection_epc_requests(
                plan, trace_group_chains=lambda *a, **k: None, **_inputs()
            )
        )
    assert calls == []


def test_auto_hands_a_depth_over_budget_and_every_deeper_one_to_rays(monkeypatch):
    calls = []
    _recording_builder(monkeypatch, calls)
    _fake_tree(monkeypatch, [])
    # Depth 2 has 12 sequences x 5 receivers = 60 pairs; a budget of 40 sends it to rays.
    monkeypatch.setattr(reflection, "_COMPLETE_PAIR_BUDGET", 40)
    traced = []

    def trace(tx, *, face_group_id, max_depth):
        traced.append(max_depth)
        chains = torch.tensor([[0, 1, 2], [1, 1, 2], [2, 0, -1]], dtype=torch.long)
        return chains.repeat(int(tx.shape[0]), 1)

    record = reflection.ReflectionDiscoveryRecord()
    plan = reflection.prepare_reflection_discovery_plan(_groups(), min_depth=1, max_depth=3)
    list(
        reflection.iter_reflection_epc_requests(
            plan, trace_group_chains=trace, record=record, **_inputs()
        )
    )
    assert record.sources == {1: "image_tree", 2: "rays", 3: "rays"}
    assert not record.complete
    assert traced == [3]
    # Depth 2 from rays: chains [0,1] and [2,0] per transmitter; [1,1] repeats a plane group.
    depth2 = [c for c in calls if c["sequences"].shape[1] == 2]
    assert depth2[0]["sequences"].tolist() == [[10, 11], [12, 10], [10, 11], [12, 10]]
    assert depth2[0]["sequence_tx"].tolist() == [0, 0, 1, 1]


def test_rays_mode_traces_once_and_is_never_complete(monkeypatch):
    calls = []
    _recording_builder(monkeypatch, calls)
    _fake_tree(monkeypatch, expansions := [])

    def trace(tx, *, face_group_id, max_depth):
        return torch.tensor([[-1], [2], [1], [2]], dtype=torch.long).repeat(int(tx.shape[0]), 1)

    record = reflection.ReflectionDiscoveryRecord()
    plan = reflection.prepare_reflection_discovery_plan(
        _groups(), min_depth=1, max_depth=1, discovery="rays"
    )
    list(reflection.iter_reflection_epc_requests(plan, trace_group_chains=trace, record=record,
                                                 **_inputs()))
    assert expansions == []
    assert record.sources == {1: "rays"} and not record.complete
    assert calls[0]["sequences"].tolist() == [[11], [12], [11], [12]]
    assert calls[0]["sequence_tx"].tolist() == [0, 0, 1, 1]


def test_large_products_go_through_the_receiver_test(monkeypatch):
    calls = []
    _recording_builder(monkeypatch, calls)
    _fake_tree(monkeypatch, [])
    monkeypatch.setattr(reflection, "_RECEIVER_TEST_MIN_PAIRS", 0)
    monkeypatch.setattr(reflection, "_PAIR_TESTS_PER_LAUNCH", 20)
    tested = []

    def receiver_test(**kwargs):
        tested.append((kwargs["sequence_start"], kwargs["sequence_end"],
                       kwargs["min_segment_length"]))
        start = kwargs["sequence_start"]
        return torch.tensor([start], dtype=torch.int32), torch.tensor([0], dtype=torch.int32)

    monkeypatch.setattr(reflection.topology_kernels, "reflection_receiver_candidates", receiver_test)
    plan = reflection.prepare_reflection_discovery_plan(_groups(), min_depth=1, max_depth=1)
    counts = []
    list(reflection.iter_reflection_epc_requests(
        plan, trace_group_chains=lambda *a, **k: None, record_candidate_count=counts.append,
        **_inputs()))
    # Six (tx, group) sequences, four per launch at five receivers each.
    assert [t[:2] for t in tested] == [(0, 4), (4, 6)]
    assert tested[0][2] == pytest.approx(0.5 * 1.0e-5 * 100.0)
    assert counts == [1, 1]
    assert [c["candidate_sequence"].tolist() for c in calls] == [[0], [4]]


def test_unique_tx_chains_matches_per_transmitter_unique_and_skips_repeats():
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
            if depth == 2:
                reached = reached[reached[:, 0] != reached[:, 1]]
            for row in torch.unique(reached, dim=0).tolist():
                expected.append([tx, *row])
        assert rows.tolist() == expected
    wide = reflection._unique_tx_chains(chains, depth=2, group_count=2**40)
    assert wide.tolist() == reflection._unique_tx_chains(chains, depth=2, group_count=3).tolist()


def test_generator_is_lazy_and_callback_errors_propagate():
    calls = []

    def trace(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("trace failed")

    plan = reflection.prepare_reflection_discovery_plan(
        _groups(), min_depth=1, max_depth=1, discovery="rays"
    )
    iterator = reflection.iter_reflection_epc_requests(plan, trace_group_chains=trace, **_inputs())
    assert calls == []
    with pytest.raises(RuntimeError, match="trace failed"):
        next(iterator)


def test_no_group_yields_nothing():
    plan = reflection.prepare_reflection_discovery_plan(_groups(0), min_depth=1, max_depth=2)
    assert list(reflection.iter_reflection_epc_requests(
        plan, trace_group_chains=lambda *a, **k: None, **_inputs())) == []


def test_contract_fields_and_forbidden_imports():
    assert is_dataclass(reflection.ReflectionDiscoveryPlan)
    assert reflection.ReflectionDiscoveryPlan.__dataclass_params__.frozen
    assert [field.name for field in fields(reflection.ReflectionDiscoveryPlan)] == [
        "groups",
        "min_depth",
        "max_depth",
        "discovery",
    ]
    assert [field.name for field in fields(reflection.ReflectionEpcRequest)] == [
        "depth",
        "epc_inputs",
    ]
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
        "witwin.channel.components",
        "witwin.channel.materials",
        "witwin.channel.kernels",
        "witwin.channel.kernels.topology",
        "witwin.channel.propagation.geometry",
        "witwin.channel.propagation.topology",
        "witwin.channel.scene.compiler",
        "witwin.channel.scene.endpoints",
        "witwin.channel.scene.resources",
    }
    assert not any(module.startswith("witwin.channel.core") for module in imports)
