# Copyright Xingyu Chen.
# Tests evaluation contract.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from witwin.channel import materials as materials_package
from witwin.channel import radiomap
from witwin.channel.radiomap import _stochastic as mc_basic
from witwin.channel import runtime


def test_frequency_and_material_guards_have_canonical_owners():
    frequency_guard = runtime._frequency_participates_in_ad
    material_guard = materials_package._require_frequency_ad_constant_materials

    assert radiomap._stochastic._require_frequency_ad_constant_materials is material_guard
    assert frequency_guard.__module__ == runtime.__name__
    assert material_guard.__module__ == materials_package.__name__
    assert (
        runtime._participates_in_ad is runtime._ad_geometry_live
    )
    assert "_require_frequency_ad_constant_materials" not in getattr(materials_package, "__all__", ())


def test_frequency_participation_detects_reverse_and_forward_mode():
    participates = runtime._frequency_participates_in_ad

    assert not participates(3.0e9)
    assert not participates(torch.tensor(3.0e9))
    assert participates(torch.tensor(3.0e9, requires_grad=True))

    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(3.0e9), torch.ones(()))
        assert participates(dual)


def _compiled_with_dependencies(*names: str) -> SimpleNamespace:
    return SimpleNamespace(materials=SimpleNamespace(frequency_dependent=tuple(names)))


def test_material_guard_allows_constant_materials_and_non_ad_frequency():
    guard = materials_package._require_frequency_ad_constant_materials

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
        materials_package._require_frequency_ad_constant_materials(
            SimpleNamespace(frequency=torch.tensor(3.0e9, requires_grad=True)),
            _compiled_with_dependencies("debye-wall"),
            ad_mode="vjp",
        )


def test_material_guard_rejects_dependent_materials_for_forward_mode_frequency():
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(3.0e9), torch.ones(()))
        with pytest.raises(NotImplementedError, match="debye-wall"):
            materials_package._require_frequency_ad_constant_materials(
                SimpleNamespace(frequency=dual),
                _compiled_with_dependencies("debye-wall"),
                ad_mode="jvp",
            )


def test_mc_frequency_material_guard_runs_before_build_info_and_native(monkeypatch):
    from witwin.core import Scene

    events: list[str] = []
    compiled = object()
    bound = SimpleNamespace(transmitters=(), receivers=(), frequency=3.0e9)

    config = radiomap.Stochastic(
        workspace_limit_bytes=None,
        components={"los"},
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
        mc_basic, "validate_scalar_endpoint_features", lambda *a, **k: None
    )
    monkeypatch.setattr(mc_basic, "_endpoint_views", lambda scene: ())
    monkeypatch.setattr(
        mc_basic, "_validate_scalar_endpoint_boundary", lambda views: None
    )
    monkeypatch.setattr(
        mc_basic,
        "compile_scene",
        lambda *args, **kwargs: events.append("compile") or compiled,
    )
    monkeypatch.setattr(mc_basic, "bind_solver_scene", lambda value: bound)
    monkeypatch.setattr(mc_basic, "require_compiled", lambda scene: compiled)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mc_basic, "_require_frequency_ad_constant_materials", guard)
    monkeypatch.setattr(mc_basic, "build_info", forbidden("build_info"))
    monkeypatch.setattr(mc_basic, "make_cuda_generator", forbidden("native"))

    with pytest.raises(NotImplementedError, match="frequency material guard"):
        radiomap.solve(Scene(), config, reference_frequency_hz=3.0e9)

    assert events == ["compile", "guard"]
