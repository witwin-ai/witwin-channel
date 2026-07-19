from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel_native.propagation.enumerated import transmission
from witwin.channel_native.propagation.geometry import (
    transmission as geometry_transmission,
)


class _Rayd:
    available = True

    def __init__(self, events):
        self.events = events

    def require_resource(self):
        self.events.append("handle")
        return 41

    def edge_records(self):
        self.events.append("records")
        return SimpleNamespace(
            vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
        )


def _hit_result(
    t: list[float],
    position: list[list[float]],
    primitive: list[int],
) -> geometry_transmission.TransmissionClosestHitResult:
    return geometry_transmission.TransmissionClosestHitResult(
        t=torch.tensor(t, dtype=torch.float32),
        position=torch.tensor(position, dtype=torch.float32),
        geometric_normal=torch.ones((len(t), 3), dtype=torch.float32),
        global_primitive_id=torch.tensor(primitive, dtype=torch.int32),
    )


def test_transmission_fake_march_preserves_events_guards_offsets_and_padding(
    monkeypatch,
):
    events: list[str] = []
    rayd = _Rayd(events)
    compiled = SimpleNamespace(
        rayd=rayd,
        assignments=SimpleNamespace(
            face_material_id=torch.tensor([0, 1], dtype=torch.int64)
        ),
        materials=SimpleNamespace(
            geometry_mode_id=torch.tensor([0, 1], dtype=torch.int64)
        ),
    )
    scene = SimpleNamespace(structures=[object()])
    tx_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    rx_positions = torch.tensor(
        [[5.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=torch.float32
    )

    original_prepare = transmission.prepare_transmission_pair_plan
    original_iter = transmission.iter_transmission_active_rows
    original_select = transmission.select_transmission_winner_rows
    original_ensure = transmission._ensure_topology_fields

    def tracked_prepare(**kwargs):
        events.append("prepare")
        return original_prepare(**kwargs)

    def tracked_iter(plan, *, done, invalid):
        events.append("iterator")
        for request in original_iter(plan, done=done, invalid=invalid):
            events.append(f"yield:{request.step}")
            yield request
            events.append(f"resume:{request.step}")

    def tracked_select(**kwargs):
        events.append("winner")
        assert kwargs["bad_material"].tolist() == [False, False, True, False]
        return original_select(**kwargs)

    def tracked_ensure(*args, **kwargs):
        events.append("ensure")
        return original_ensure(*args, **kwargs)

    def fake_normalize(values, *, eps):
        events.append(f"normalize:{values.shape[0]}")
        assert values.is_contiguous()
        assert eps == 1.0e-9
        return (values / values.norm(dim=-1).clamp_min(eps)[:, None]).contiguous()

    responses = [
        _hit_result(
            [2.0, float("inf"), 2.0, 2.0],
            [
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.6, 0.0],
                [2.0, 1.0, 0.0],
            ],
            [0, -1, 1, 0],
        ),
        _hit_result(
            [float("inf"), float("inf"), 1.0],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
            [-1, -1, 0],
        ),
        _hit_result([0.5], [[4.0, 1.0, 0.0]], [0]),
    ]
    queries = []

    def fake_query(query):
        index = len(queries)
        events.append(f"query:{index}")
        assert isinstance(query, geometry_transmission.TransmissionClosestHitQuery)
        assert query.handle == 41
        assert query.active is None
        assert query.flags == 7
        assert query.origin.is_contiguous()
        assert query.direction.is_contiguous()
        assert query.ray_tmax.is_contiguous()
        queries.append(query)
        return responses[index]

    monkeypatch.setattr(
        transmission, "prepare_transmission_pair_plan", tracked_prepare
    )
    monkeypatch.setattr(transmission, "iter_transmission_active_rows", tracked_iter)
    monkeypatch.setattr(
        transmission, "select_transmission_winner_rows", tracked_select
    )
    monkeypatch.setattr(transmission, "_ensure_topology_fields", tracked_ensure)
    monkeypatch.setattr(
        transmission.geometry_primitives,
        "deterministic_normalize_vec3",
        fake_normalize,
    )
    monkeypatch.setattr(
        transmission,
        "query_transmission_closest_hit",
        fake_query,
    )

    block, launch_count, candidate_count, guardrail_count = (
        transmission._transmission_topology(
            scene,
            compiled,
            tx_positions,
            rx_positions,
            max_depth=2,
        )
    )

    assert (launch_count, candidate_count, guardrail_count) == (3, 3, 2)
    assert events == [
        "handle",
        "records",
        "prepare",
        "normalize:4",
        "iterator",
        "yield:0",
        "query:0",
        "normalize:3",
        "resume:0",
        "yield:1",
        "query:1",
        "normalize:1",
        "resume:1",
        "yield:2",
        "query:2",
        "resume:2",
        "winner",
        "ensure",
    ]
    assert [query.origin.shape[0] for query in queries] == [4, 3, 1]
    assert float(queries[1].origin[0, 0]) == pytest.approx(2.00001)
    assert float(queries[2].origin[0, 0]) == pytest.approx(3.00001)
    assert block["tx_id"].tolist() == [0]
    assert block["rx_id"].tolist() == [0]
    assert block["depth"].tolist() == [1]
    assert block["component_id"].tolist() == [5]
    assert block["primitive_id"].tolist() == [0]
    assert block["material_id"].tolist() == [0]
    assert block["primitive_sequence"].tolist() == [[0, -1]]
    assert block["material_sequence"].tolist() == [[0, -1]]
    assert block["interaction_positions"][0, 1].tolist() == [0.0, 0.0, 0.0]
    assert block["interaction_normals"][0, 1].tolist() == [0.0, 0.0, 0.0]
    assert block["path_length_m"].tolist() == [5.0]
