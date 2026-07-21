from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from witwin.channel_native import Scene
from witwin.channel_native.core import scene as legacy_scene
from witwin.channel_native.scene import compile as canonical_compile


_HELPER_BODY_HASHES = {
    "_compile_geometry": (
        "5c6b1722f0618151b416c259f961b2e41e8b40f1441fd0a820635688f1cdd1b1"
    ),
    "_abi_v3_layer_view": (
        "dc7258913a12cb41c84af8446726e1c656e41c32e8fee332c1025bffd6b2e23e"
    ),
    "_phase_screen_descriptor": (
        "77ca2e0d59182862221f5e86648321d93782ba04c2d36b7b075884d8aa14a268"
    ),
    "_material_records": (
        "0496c3163d9e61f35d521d74b0f64c1a569c8c23d7da5668c339776ae6250904"
    ),
    "_frequency_dependent_material_keys": (
        "56819d7f8e2de0f2d7ac7cb16b4367d9f035dd361f2464320a492edebb8ef055"
    ),
    "_compile_materials": (
        "64b2c053c67c002ce5a2daa9ca6af4414b0db93057af6b1fbcddf9dc7e86350e"
    ),
    "_compile_assignments": (
        "936bda87572a5dd033d1b9eba9c82f39a136ff4b81a0ac3c49c741f30c8f6cf3"
    ),
}
_CANONICAL_ONLY_HELPER_BODY_HASHES = {
    "_compile_penetration_scene_diagonals": (
        "2d130cd4c4e7dce600e7090d010b7a2542551a5c0b12d41edefb09380e6109bb"
    ),
}


def _body_hash(function) -> str:
    node = ast.parse(inspect.getsource(function)).body[0]
    body = ast.Module(body=node.body, type_ignores=[])
    return hashlib.sha256(
        ast.dump(body, include_attributes=False).encode("utf-8")
    ).hexdigest()


def test_compile_helpers_have_one_exact_canonical_owner():
    for name, expected_hash in _HELPER_BODY_HASHES.items():
        owner = getattr(canonical_compile, name)
        assert getattr(legacy_scene, name) is owner
        assert owner.__module__ == canonical_compile.__name__
        assert _body_hash(owner) == expected_hash

    for name, expected_hash in _CANONICAL_ONLY_HELPER_BODY_HASHES.items():
        owner = getattr(canonical_compile, name)
        assert not hasattr(legacy_scene, name)
        assert owner.__module__ == canonical_compile.__name__
        assert _body_hash(owner) == expected_hash

    legacy_tree = ast.parse(inspect.getsource(legacy_scene))
    legacy_definitions = {
        node.name for node in legacy_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert legacy_definitions.isdisjoint(
        _HELPER_BODY_HASHES | _CANONICAL_ONLY_HELPER_BODY_HASHES
    )
    assert canonical_compile.compile_scene.__module__ == canonical_compile.__name__


def test_compile_owner_has_no_scene_or_solver_dependency_cycle():
    source_path = Path(canonical_compile.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    targets = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "witwin.channel_native.core.scene" not in targets
    assert not any(".solver" in target for target in targets)
    assert not any(
        target.startswith("witwin.channel_native.scattering") for target in targets
    )


def test_compile_cache_hit_preserves_the_exact_call_ledger(monkeypatch):
    ledger: list[str] = []
    helper_names = (
        "_material_records",
        "_compile_geometry",
        "_frequency_dependent_material_keys",
        "_compile_materials",
        "_compile_assignments",
    )
    for name in helper_names:
        original = getattr(canonical_compile, name)

        def record(*args, _name=name, _original=original, **kwargs):
            ledger.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(canonical_compile, name, record)

    original_rayd_scene = Scene.rayd_scene

    def rayd_scene(scene):
        ledger.append("rayd_scene")
        return original_rayd_scene(scene)

    monkeypatch.setattr(Scene, "rayd_scene", rayd_scene)
    scene = Scene(structures=[], transmitters=[], receivers=[], frequency=3.5e9)

    compiled = scene.compile()
    assert ledger == [
        "_material_records",
        "rayd_scene",
        "_compile_geometry",
        "_frequency_dependent_material_keys",
        "_compile_materials",
        "_compile_assignments",
    ]

    ledger.clear()
    assert scene.compile() is compiled
    assert ledger == ["_material_records"]


def test_frequency_dependency_probe_treats_value_error_as_dependent():
    class ProbeMaterial:
        def parameters(self, frequency_hz):
            raise ValueError(f"probe frequency {frequency_hz} is out of range")

    structures = (SimpleNamespace(material=ProbeMaterial()),)

    assert canonical_compile._frequency_dependent_material_keys(
        structures,
        [{"name": "probe"}],
        ("0:probe",),
        1.0e9,
    ) == ("0:probe",)


@pytest.mark.parametrize("exception_type", (TypeError, RuntimeError))
def test_frequency_dependency_probe_propagates_unexpected_errors(exception_type):
    class ProbeMaterial:
        def parameters(self, frequency_hz):
            raise exception_type(f"unexpected probe failure at {frequency_hz}")

    structures = (SimpleNamespace(material=ProbeMaterial()),)

    with pytest.raises(exception_type, match="unexpected probe failure"):
        canonical_compile._frequency_dependent_material_keys(
            structures,
            [{"name": "probe"}],
            ("0:probe",),
            1.0e9,
        )
