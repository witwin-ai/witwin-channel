from __future__ import annotations

import ast
import inspect

import pytest
import torch

from witwin.channel_native.core.kernels import ops
from witwin.channel_native.runtime import symbols
from witwin.channel_native.scene import kernels
from witwin.channel_native.scene.kernels import rayd_scene


_SCENE_KERNEL_NAMES = (
    "_raydn_module_handle",
    "_raydn_scene_handle_id",
    "raydn_scene_create",
    "raydn_scene_edge_records",
)


def _body_ast(function: object) -> str:
    node = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)


@pytest.mark.parametrize("name", _SCENE_KERNEL_NAMES)
def test_scene_kernel_body_matches_compatibility_facade(name: str):
    assert _body_ast(getattr(rayd_scene, name)) == _body_ast(getattr(ops, name))


@pytest.mark.parametrize("name", _SCENE_KERNEL_NAMES)
def test_scene_kernel_package_reexports_canonical_owner(name: str):
    owner = getattr(rayd_scene, name)

    assert owner.__module__ == rayd_scene.__name__
    assert getattr(kernels, name) is owner


def test_scene_kernel_uses_canonical_required_symbol():
    assert rayd_scene._required_native_op is symbols.required_symbol


def test_scene_create_preserves_native_argument_order(monkeypatch: pytest.MonkeyPatch):
    captured: list[tuple[object, ...]] = []
    owner = object()

    def required_symbol(name: str):
        assert name == "raydn_scene_create"

        def create(*args: object) -> tuple[int, object]:
            captured.append(args)
            return 7, owner

        return create

    monkeypatch.setattr(rayd_scene, "_required_native_op", required_symbol)
    inputs = ([torch.empty(0)] for _ in range(6))
    vertices, faces, uv, face_uv, to_world_left, to_world_right = inputs
    mesh_flags = [3]

    result = rayd_scene.raydn_scene_create(
        vertices,
        faces,
        uv,
        face_uv,
        to_world_left,
        to_world_right,
        mesh_flags,
    )

    assert result == (7, owner)
    assert captured == [
        (
            vertices,
            faces,
            uv,
            face_uv,
            to_world_left,
            to_world_right,
            mesh_flags,
            0,
        )
    ]


def test_scene_edge_records_normalizes_handle_and_tuple(
    monkeypatch: pytest.MonkeyPatch,
):
    record = torch.empty(0)
    calls: list[tuple[object, ...]] = []

    def required_symbol(name: str):
        assert name == "raydn_scene_edge_records"

        def edge_records(*args: object) -> list[torch.Tensor]:
            calls.append(args)
            return [record]

        return edge_records

    class Handle:
        def handle(self) -> int:
            return 11

    monkeypatch.setattr(rayd_scene, "_required_native_op", required_symbol)

    assert rayd_scene.raydn_scene_edge_records(Handle()) == (record,)
    assert calls == [(11, 0)]
