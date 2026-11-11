# Copyright Xingyu Chen.
# Tests compile-time mesh validation and degenerate or duplicate face removal.

from __future__ import annotations

import time

import pytest
import torch

from witwin.channel.scene.compiler import _checked_topology, _face_cleanup
from tests.support.isolated import run_isolated

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_face_cleanup_drops_repeated_index_and_duplicate_faces_in_either_winding() -> None:
    faces = torch.tensor([[0, 1, 2], [2, 1, 0], [1, 2, 0], [0, 0, 1], [3, 3, 3], [1, 3, 2]])
    keep, degenerate, duplicate = _face_cleanup("structure 0", faces, 4)
    assert keep is not None
    assert keep.tolist() == [True, False, False, False, False, True]
    assert (degenerate, duplicate) == (2, 2)


def test_face_cleanup_keeps_a_clean_mesh_untouched() -> None:
    assert _face_cleanup("structure 0", torch.tensor([[0, 1, 2], [1, 3, 2]]), 4) == (None, 0, 0)


def test_face_cleanup_uses_the_wide_key_path_for_huge_vertex_counts() -> None:
    faces = torch.tensor([[0, 1, 2], [2, 0, 1], [5, 5, 6]])
    keep, degenerate, duplicate = _face_cleanup("structure 0", faces, 2**22)
    assert keep is not None and keep.tolist() == [True, False, False]
    assert (degenerate, duplicate) == (1, 1)


def test_face_cleanup_refuses_an_edge_fan_beyond_the_edge_topology_limit() -> None:
    count = 300
    faces = torch.stack(
        (torch.zeros(count, dtype=torch.long), torch.ones(count, dtype=torch.long),
         torch.arange(count) + 2),
        dim=1,
    )
    with pytest.raises(ValueError, match=r"structure 3: the edge between vertices 0 and 1 is shared by 300 distinct faces; at most 256"):
        _face_cleanup("structure 3", faces, count + 2)


def test_face_cleanup_refuses_a_mesh_with_no_face_left() -> None:
    with pytest.raises(ValueError, match="every face repeats a vertex index"):
        _face_cleanup("structure 0", torch.tensor([[0, 0, 0], [1, 1, 2]]), 3)


@pytest.mark.parametrize(
    ("faces", "message"),
    [
        ([[0, 1, -1]], r"faces\[0\] holds index -1, outside the valid range \[0, 3\)"),
        ([[0, 1, 3]], r"faces\[0\] holds index 3, outside the valid range \[0, 3\)"),
        ([[0, 1, 2], [0, 1, 2**32 + 2]], r"faces\[1\] holds index 4294967298"),
        ([[0, 1, 2**31 - 1]], r"faces\[0\] holds index 2147483647"),
    ],
)
def test_index_range_is_checked_before_native_work(faces, message) -> None:
    with pytest.raises(ValueError, match=message):
        _checked_topology("structure 1", torch.tensor(faces), 3, None, None)


def test_face_uv_index_range_is_checked() -> None:
    with pytest.raises(ValueError, match=r"structure 0: face_uv\[0\] holds index 4, outside the valid range \[0, 4\) of its 4 uv rows"):
        _checked_topology("structure 0", torch.tensor([[0, 1, 2]]), 3, torch.tensor([[0, 1, 4]]), 4)


def test_floating_point_face_tables_are_refused() -> None:
    with pytest.raises(TypeError, match="faces must be an integer tensor"):
        _checked_topology("structure 0", torch.tensor([[0.0, 1.0, 2.0]]), 3, None, None)


def test_large_clean_mesh_check_is_cheap() -> None:
    side = 400
    index = (torch.arange(side)[:, None] * (side + 1) + torch.arange(side)[None, :]).reshape(-1)
    faces = torch.cat(
        (torch.stack((index, index + 1, index + side + 1), 1),
         torch.stack((index + 1, index + side + 2, index + side + 1), 1))
    )
    start = time.process_time()
    assert _face_cleanup("structure 0", faces, (side + 1) ** 2) == (None, 0, 0)
    assert time.process_time() - start < 5.0


@CUDA
def test_bad_meshes_raise_named_errors_and_leave_the_context_usable() -> None:
    outcomes = run_isolated(
        """
        T = torch.tensor
        TRI = [[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, 0.0, 3.0]]
        def bad(name, structure):
            endpoints = [tx(), grid()] if name == "basic" else None
            return lambda: solve(name, scene([floor(), structure], endpoints))
        for name in ("det", "path", "basic", "bdpt"):
            case(f"negative/{name}", bad(name, mesh(T(TRI), T([[0, 1, -1]]))))
            case(f"equal_v/{name}", bad(name, mesh(T(TRI), T([[0, 1, 3]]))))
            case(f"int32max/{name}", bad(name, mesh(T(TRI), T([[0, 1, 2**31 - 1]]))))
            case(f"wrap/{name}", bad(name, mesh(T(TRI), T([[0, 1, 2**32 + 2]]))))
            case(f"cuda/{name}", bad(name, mesh(T(TRI).cuda(), T([[0, 1, 7]]).cuda())))
            case(f"nan/{name}", bad(name, mesh(T([[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [float("nan"), 0.0, 3.0]]), T([[0, 1, 2]]))))
        """
    )
    assert outcomes.pop("__health__") == {"status": "returned", "value": True}
    for name, outcome in outcomes.items():
        assert outcome["status"] == "raised" and outcome["type"] == "ValueError", (name, outcome)
        assert str(outcome["message"]).startswith("structure 1: "), (name, outcome)
        if name.startswith("nan/"):
            assert "vertices[2] is not finite" in str(outcome["message"])
        else:
            assert "outside the valid range [0, 3)" in str(outcome["message"])


@CUDA
def test_duplicate_and_padding_faces_compile_quickly_and_match_the_clean_mesh() -> None:
    outcomes = run_isolated(
        """
        import time
        from witwin.channel.scene import compile
        T = torch.tensor
        V = T([[2.5, -2.0, -1.0], [2.5, 2.0, -1.0], [2.5, -2.0, 3.0], [2.5, 2.0, 3.0]])
        QUAD = T([[0, 1, 2], [1, 3, 2]])
        def run(faces):
            value = scene([floor(), mesh(V, faces)])
            start = time.perf_counter()
            compiled = compile(value, reference_frequency_hz=3.5e9)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            result = solve("det", value)
            return {"compile_s": elapsed, "dropped": compiled.dropped_faces,
                    "faces": int(compiled.geometry.faces.shape[0]),
                    "gain": float(result.path_gain.sum()), "metadata": result.metadata.get("dropped_faces")}
        case("clean", lambda: run(QUAD))
        case("dup1000", lambda: run(QUAD.repeat(1000, 1)))
        case("pad1000", lambda: run(torch.cat((QUAD, T([[0, 0, 0]]).repeat(1000, 1)))))
        """
    )
    assert outcomes.pop("__health__")["value"] is True
    clean = outcomes["clean"]["value"]
    assert clean["dropped"] == {"degenerate": 0, "duplicate": 0}
    assert clean["metadata"] is None
    for name, expected in (("dup1000", {"degenerate": 0, "duplicate": 1998}),
                           ("pad1000", {"degenerate": 1000, "duplicate": 0})):
        value = outcomes[name]["value"]
        assert value["dropped"] == expected
        assert value["metadata"] == expected
        assert value["faces"] == clean["faces"]
        assert value["gain"] == clean["gain"]
        assert value["compile_s"] < 1.0, value


@CUDA
def test_primitive_ids_follow_the_kept_faces() -> None:
    from witwin.core import Mesh, PhysicalMaterial, Scene, Structure

    from witwin.channel.scene import compile

    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2], [0, 0, 1], [2, 1, 0], [1, 3, 2]])
    structure = Structure(
        Mesh(vertices, faces, recenter=False, fill_mode="surface", topology_diagnostics=False),
        PhysicalMaterial(eps_r=4.0),
        primitive_ids=(11, 12, 13, 14),
    )
    compiled = compile(Scene(structures=[structure]), reference_frequency_hz=3.5e9)
    assert compiled.geometry.face_primitive_id.tolist() == [11, 14]
    assert compiled.dropped_faces == {"degenerate": 1, "duplicate": 1}
