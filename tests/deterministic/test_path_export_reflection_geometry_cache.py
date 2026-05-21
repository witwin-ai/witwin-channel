from __future__ import annotations

import pytest

from witwin.channel.deterministic import types as wt
from witwin.channel.deterministic.reflection.detail import SourcePathSet, TraceDetail
from witwin.channel.deterministic.trace import path_export


def _point(xs, ys, zs) -> wt.Point3f:
    return wt.Point3f(xs, ys, zs)


def _source_paths(depth: int) -> SourcePathSet:
    return SourcePathSet(
        image_source=_point([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]),
        discovery_count=wt.UInt32([1, 1]),
        chain_depth=depth,
        n_paths=2,
        path_prim_idx=tuple(wt.Int32([0, 0]) for _ in range(depth)),
        path_plane_point=tuple(_point([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]) for _ in range(depth)),
        path_plane_normal=tuple(_point([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]) for _ in range(depth)),
        path_hit_point=tuple(_point([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]) for _ in range(depth)),
    )


@pytest.mark.gpu
def test_cached_reflection_geometry_refs_materialize_without_epc_replay(monkeypatch):
    def fail_epc_replay(*args, **kwargs):
        raise AssertionError("cached reflection geometry must not replay EPC during assembly")

    monkeypatch.setattr(path_export, "epc_reflection_chain_to_target", fail_epc_replay)

    raw = {
        "payload_kind": path_export._REFLECTION_PATH_REFS_PAYLOAD,
        "rx_index": wt.UInt32([0, 1]),
        "tx_index": wt.UInt32([0, 0]),
        "path_group_index": wt.UInt32([0, 1]),
        "path_idx": wt.UInt32([0, 1]),
        "a": wt.Complex2f([1.0, 2.0], [0.0, 0.0]),
        "tau": wt.Float([1.0, 2.0]),
        "tx_pos": _point([0.0], [0.0], [0.0]),
        "rx_positions": _point([10.0, 20.0], [0.0, 0.0], [0.0, 0.0]),
        "scene": object(),
        "reflection_detail": TraceDetail(
            reflection_model="materialized",
            reflection_model_source="test",
            reflection_gain=1.0,
            source_paths_per_bounce=(_source_paths(1), _source_paths(2)),
        ),
        "wavelength": 1.0,
        "tx_polarization": (1.0, 0.0, 0.0),
        "max_depth_hint": 2,
        "theta_t": wt.Float([0.1, 0.2]),
        "phi_t": wt.Float([0.3, 0.4]),
        "theta_r": wt.Float([0.5, 0.6]),
        "phi_r": wt.Float([0.7, 0.8]),
        "path_depth": wt.UInt32([2, 1]),
        "type_slots": (
            wt.Int32([path_export.InteractionType.REFLECTION, path_export.InteractionType.REFLECTION]),
            wt.Int32([path_export.InteractionType.REFLECTION, path_export.InteractionType.NONE]),
        ),
        "vertex_slots": (
            _point([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]),
            _point([7.0, 0.0], [8.0, 0.0], [9.0, 0.0]),
        ),
        "normal_slots": (
            _point([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]),
            _point([0.0, 0.0], [0.0, 0.0], [1.0, 0.0]),
        ),
        "object_slots": (
            wt.Int32([11, 12]),
            wt.Int32([13, -1]),
        ),
        "metadata": {"n_paths": 2},
    }

    materialized = path_export._materialize_reflection_path_refs(
        raw,
        return_geometry=True,
        path_indices=wt.UInt32([0]),
    )

    assert materialized["payload_kind"] == path_export._MATERIALIZED_PATH_PAYLOAD
    assert int(materialized["rx_index"][0]) == 0
    assert int(materialized["type_slots"][1][0]) == path_export.InteractionType.REFLECTION
    assert float(materialized["vertex_slots"][1].x[0]) == pytest.approx(7.0)
    assert int(materialized["object_slots"][1][0]) == 13
