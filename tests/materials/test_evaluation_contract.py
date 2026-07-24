from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from witwin.channel import materials as materials_package
from witwin.channel.materials import evaluation
from witwin.channel.montecarlo.basic import solver as mc_solver
from witwin.channel.runtime import autograd_contracts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

_FUNCTION_DIGESTS = {
    "_frequency_participates_in_ad": (
        "2f734599df63cffb659b87efbca9aca37afe31925d9bfaefa7ae085ec73470d9"
    ),
    "_require_frequency_ad_constant_materials": (
        "6fcb1f433cd7f4be0912081266731dd84eb810be54f64feff6a68243915b79d4"
    ),
}


def _function_digest(module, name: str) -> str:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    payload = ast.dump(definition, include_attributes=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_frequency_and_material_guards_have_canonical_owners():
    frequency_guard = autograd_contracts._frequency_participates_in_ad
    material_guard = evaluation._require_frequency_ad_constant_materials

    assert mc_solver._require_frequency_ad_constant_materials is material_guard
    assert frequency_guard.__module__ == autograd_contracts.__name__
    assert material_guard.__module__ == evaluation.__name__
    assert (
        autograd_contracts._participates_in_ad is autograd_contracts._ad_geometry_live
    )
    assert "_require_frequency_ad_constant_materials" not in materials_package.__all__


def test_moved_function_bodies_and_signatures_are_exact():
    assert (
        _function_digest(autograd_contracts, "_frequency_participates_in_ad")
        == _FUNCTION_DIGESTS["_frequency_participates_in_ad"]
    )
    assert (
        _function_digest(evaluation, "_require_frequency_ad_constant_materials")
        == _FUNCTION_DIGESTS["_require_frequency_ad_constant_materials"]
    )


def test_frequency_participation_detects_reverse_and_forward_mode():
    participates = autograd_contracts._frequency_participates_in_ad

    assert not participates(3.0e9)
    assert not participates(torch.tensor(3.0e9))
    assert participates(torch.tensor(3.0e9, requires_grad=True))

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(3.0e9), torch.ones(()))
        assert participates(dual)


def _compiled_with_dependencies(*names: str) -> SimpleNamespace:
    return SimpleNamespace(materials=SimpleNamespace(frequency_dependent=tuple(names)))


def test_material_guard_allows_constant_materials_and_non_ad_frequency():
    guard = evaluation._require_frequency_ad_constant_materials

    guard(
        SimpleNamespace(frequency=torch.tensor(3.0e9, requires_grad=True)),
        _compiled_with_dependencies(),
        ad_mode="vjp",
    )
    guard(
        SimpleNamespace(frequency=torch.tensor(3.0e9)),
        _compiled_with_dependencies("debye-wall"),
        ad_mode="vjp",
    )


def test_material_guard_rejects_dependent_materials_for_reverse_mode_frequency():
    with pytest.raises(NotImplementedError, match="debye-wall"):
        evaluation._require_frequency_ad_constant_materials(
            SimpleNamespace(frequency=torch.tensor(3.0e9, requires_grad=True)),
            _compiled_with_dependencies("debye-wall"),
            ad_mode="vjp",
        )


def test_material_guard_rejects_dependent_materials_for_forward_mode_frequency():
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(3.0e9), torch.ones(()))
        with pytest.raises(NotImplementedError, match="debye-wall"):
            evaluation._require_frequency_ad_constant_materials(
                SimpleNamespace(frequency=dual),
                _compiled_with_dependencies("debye-wall"),
                ad_mode="jvp",
            )


def test_mc_frequency_material_guard_runs_before_build_info_and_native(monkeypatch):
    from witwin.core import Scene
    from witwin.channel.montecarlo.basic import pipeline as mc_pipeline

    events: list[str] = []
    compiled = object()
    bound = SimpleNamespace(transmitters=(), receivers=(), frequency=3.0e9)

    config = SimpleNamespace(
        workspace_limit_bytes=None,
        components=frozenset(),
        max_depth=1,
        ad_mode="vjp",
    )

    def guard(scene, actual_compiled, *, ad_mode: str):
        assert scene is bound
        assert actual_compiled is compiled
        assert ad_mode == "vjp"
        events.append("guard")
        raise NotImplementedError("frequency material guard")

    def forbidden(name: str):
        def call(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} ran before the material guard")

        return call

    monkeypatch.setattr(
        mc_solver, "validate_scalar_endpoint_features", lambda *a, **k: None
    )
    monkeypatch.setattr(mc_solver, "_endpoint_views", lambda scene: ())
    monkeypatch.setattr(
        mc_solver, "_validate_scalar_endpoint_boundary", lambda views: None
    )
    monkeypatch.setattr(
        mc_solver,
        "compile_scene",
        lambda *args, **kwargs: events.append("compile") or compiled,
    )
    monkeypatch.setattr(mc_solver, "bind_solver_scene", lambda value: bound)
    monkeypatch.setattr(mc_pipeline, "require_compiled", lambda scene: compiled)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mc_solver, "_require_frequency_ad_constant_materials", guard)
    monkeypatch.setattr(mc_solver, "build_info", forbidden("build_info"))
    monkeypatch.setattr(mc_solver, "make_cuda_generator", forbidden("native"))

    with pytest.raises(NotImplementedError, match="frequency material guard"):
        mc_solver.solve(Scene(), config, reference_frequency_hz=3.0e9)

    assert events == ["compile", "guard"]
