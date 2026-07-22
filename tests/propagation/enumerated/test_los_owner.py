from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ci import check_import_graph as graph
from witwin.channel.propagation.enumerated import los
from witwin.channel.propagation.geometry import visibility


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "witwin" / "channel_native"


def _fake_exported() -> dict[str, torch.Tensor]:
    return {
        "tx_id": torch.tensor([0, 1], dtype=torch.int32),
        "rx_id": torch.tensor([1, 0], dtype=torch.int32),
        "path_length_m": torch.tensor([2.0, 3.0], dtype=torch.float32),
        "delay_s": torch.tensor([0.2, 0.3], dtype=torch.float32),
        "path_gain": torch.tensor([0.5, 0.25], dtype=torch.float32),
    }


def test_los_and_visibility_canonical_owner_identity():
    assert los._los_topology.__module__ == los.__name__
    assert visibility._rayd_visibility_mask.__module__ == visibility.__name__
    assert visibility._los_visibility_mask.__module__ == visibility.__name__


def test_los_topology_preserves_fake_event_and_count_semantics(monkeypatch):
    events = []
    exported = _fake_exported()
    tx_positions = torch.zeros((2, 3), dtype=torch.float32)
    tx_power = torch.ones((2,), dtype=torch.float32)
    rx_positions = torch.ones((2, 3), dtype=torch.float32)
    tx_polarizations = torch.zeros((2, 3), dtype=torch.float32)
    start = tx_positions.clone()
    end = rx_positions.clone()
    active = torch.tensor([True, True])
    visible = torch.tensor([True, False])
    raw_block = {"valid": torch.tensor([True])}
    original_prepare = los.prepare_los_candidates

    def fake_export(*args, **kwargs):
        events.append("export")
        # R5: path_los_export now also takes the per-transmitter polarization.
        assert len(args) == 4
        assert all(
            actual is expected
            for actual, expected in zip(
                args,
                (tx_positions, tx_power, rx_positions, tx_polarizations),
                strict=True,
            )
        )
        assert kwargs == {"frequency_hz": 3.0e9}
        return exported

    def fake_prepare(**kwargs):
        events.append("plan")
        return original_prepare(**kwargs)

    def fake_visibility_inputs(*args):
        events.append("visibility_inputs")
        assert args[0] is tx_positions
        assert args[1] is rx_positions
        assert args[2].data_ptr() == exported["tx_id"].data_ptr()
        assert args[3].data_ptr() == exported["rx_id"].data_ptr()
        assert args[2].is_contiguous() and args[3].is_contiguous()
        return {"start": start, "end": end, "active": active}

    def fake_visibility(query):
        events.append("visibility")
        assert isinstance(query, visibility.VisibilityQuery)
        assert query.rayd is compiled.rayd
        assert query.start is start
        assert query.end is end
        assert query.active is active
        return visibility.VisibilityResult(visible=visible)

    def fake_construct(*args, **kwargs):
        events.append("construct")
        assert args[0].data_ptr() == exported["tx_id"].data_ptr()
        assert args[1].data_ptr() == exported["rx_id"].data_ptr()
        assert args[2].data_ptr() == exported["path_length_m"].data_ptr()
        assert args[3].data_ptr() == exported["delay_s"].data_ptr()
        assert args[4].data_ptr() == exported["path_gain"].data_ptr()
        assert args[5] is visible
        assert kwargs == {"frequency_hz": 3.0e9, "sequence_width": 2}
        return raw_block

    def fake_ensure(block):
        events.append("ensure")
        assert block is raw_block
        return block

    compiled = SimpleNamespace(rayd=object())
    monkeypatch.setattr(los.topology_blocks, "path_los_export", fake_export)
    monkeypatch.setattr(los, "prepare_los_candidates", fake_prepare)
    monkeypatch.setattr(
        los.topology_blocks,
        "path_los_visibility_inputs",
        fake_visibility_inputs,
    )
    monkeypatch.setattr(los, "run_visibility_query", fake_visibility)
    monkeypatch.setattr(
        los.topology_construction,
        "deterministic_los_topology_block",
        fake_construct,
    )
    monkeypatch.setattr(los, "_ensure_topology_fields", fake_ensure)

    result = los._los_topology(
        SimpleNamespace(structures=[object()]),
        compiled,
        tx_positions,
        tx_power,
        rx_positions,
        tx_polarizations,
        frequency_hz=3.0e9,
        sequence_width=2,
    )

    assert result == (raw_block, 2, 2, 1)
    assert events == [
        "export",
        "plan",
        "visibility_inputs",
        "visibility",
        "construct",
        "ensure",
    ]


def test_los_topology_skips_visibility_for_structure_free_scene(monkeypatch):
    exported = _fake_exported()
    raw_block = {"valid": torch.tensor([True, True])}
    monkeypatch.setattr(
        los.topology_blocks,
        "path_los_export",
        lambda *args, **kwargs: exported,
    )
    monkeypatch.setattr(
        los.topology_blocks,
        "path_los_visibility_inputs",
        lambda *args, **kwargs: pytest.fail("visibility inputs must be skipped"),
    )
    monkeypatch.setattr(
        los,
        "run_visibility_query",
        lambda *args, **kwargs: pytest.fail("visibility query must be skipped"),
    )

    def fake_construct(*args, **kwargs):
        assert args[5] is None
        return raw_block

    monkeypatch.setattr(
        los.topology_construction,
        "deterministic_los_topology_block",
        fake_construct,
    )
    monkeypatch.setattr(los, "_ensure_topology_fields", lambda block: block)

    result = los._los_topology(
        SimpleNamespace(structures=[]),
        SimpleNamespace(rayd=object()),
        torch.zeros((2, 3), dtype=torch.float32),
        torch.ones((2,), dtype=torch.float32),
        torch.ones((2, 3), dtype=torch.float32),
        torch.ones((2, 3), dtype=torch.float32),
        frequency_hz=3.0e9,
        sequence_width=0,
    )

    assert result == (raw_block, 1, 2, 0)


def test_los_owners_have_no_core_path_dependency_or_scc():
    owners = {
        "witwin.channel.propagation.enumerated.los",
        "witwin.channel.propagation.geometry.visibility",
        "witwin.channel.propagation.topology.discovery.los",
    }
    core = "witwin.channel.core.path_topology"
    edges = graph.collect_import_edges(PACKAGE_ROOT)
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
    for owner in owners:
        assert core not in adjacency.get(owner, set())
        pending = [owner]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(adjacency.get(current, ()))
        assert core not in seen


def test_enumerated_public_all_is_unchanged():
    import witwin.channel.propagation.enumerated as enumerated

    assert enumerated.__all__ == []
