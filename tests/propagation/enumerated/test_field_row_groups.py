# Copyright Xingyu Chen.
# Tests the compact owner's field-stage row groups and the sync-free field stage.

from __future__ import annotations

import pytest
import torch

from tests.deterministic.test_reflection_multibounce import two_wall_multibounce_scene
from tests.support.scenes import empty_space_los_scene
from witwin.channel.deployment import build_info
from witwin.channel.deterministic import Config
from witwin.channel.propagation import fields as field_stage
from witwin.channel.propagation.enumerated import evaluate_enumerated_paths
from witwin.channel.propagation.rows import COMPONENT_ROW_GROUPS
from witwin.channel.scene import compile as compile_scene
from witwin.channel.scene.endpoints import bind_solver_scene


def _require_native():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the enumerated field stage")
    if not build_info()["uses_rayd_native"]:
        pytest.skip("RayD native reflection is not built")


def _solve(scene, config):
    compiled = bind_solver_scene(compile_scene(scene, reference_frequency_hz=3.0e9))
    return evaluate_enumerated_paths(compiled, config)


def test_compact_row_groups_list_each_component_and_reflection_depth_in_row_order():
    _require_native()
    evaluated, sidecars = _solve(
        two_wall_multibounce_scene(),
        Config(components={"los", "reflection"}, max_depth=3),
    )
    groups = sidecars.row_groups
    assert groups is not None
    assert groups.row_count == evaluated.row_count > 0
    component = evaluated.topology.component_id.cpu()
    depth = evaluated.topology.depth.cpu()
    for group in range(len(groups.counts) - 1):
        if group >= COMPONENT_ROW_GROUPS:
            mask = (component == 1) & (depth == group - COMPONENT_ROW_GROUPS)
        elif group == 1:
            mask = torch.zeros_like(component, dtype=torch.bool)
        else:
            mask = component == group
        expected = torch.nonzero(mask, as_tuple=False).reshape(-1)
        assert groups.count(group) == int(expected.numel()), group
        if groups.count(group):
            rows = groups.rows(group)
            rows = torch.arange(groups.row_count) if rows is None else rows.cpu()
            assert torch.equal(rows, expected), group
    assert groups.counts[-1] == 0
    assert set(groups.reflection_depths()) == set(depth[component == 1].tolist())


def test_los_only_rows_form_one_identity_group():
    _require_native()
    evaluated, sidecars = _solve(empty_space_los_scene(), Config(max_depth=0, components={"los"}))
    assert sidecars.compact_metadata.count_d2h_copies == 0
    assert evaluated.row_count > 0
    assert bool((evaluated.topology.component_id == 0).all())


def test_field_stage_component_passes_issue_no_torch_synchronization(monkeypatch):
    _require_native()
    scene = two_wall_multibounce_scene()
    config = Config(components={"los", "reflection"}, max_depth=3)
    reference, _ = _solve(scene, config)

    def strict(function):
        def call(*args, **kwargs):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                return function(*args, **kwargs)
            finally:
                torch.cuda.set_sync_debug_mode(previous)

        return call

    for name in (
        "_evaluate_los_fields",
        "_evaluate_reflection_fields",
        "_evaluate_transmission_fields",
        "_evaluate_diffraction_fields",
        "_evaluate_coupled_fields",
    ):
        monkeypatch.setattr(field_stage, name, strict(getattr(field_stage, name)))
    evaluated, _ = _solve(scene, config)
    assert torch.equal(evaluated.fields.path_field, reference.fields.path_field)
    assert torch.equal(evaluated.geometry.path_length_m, reference.geometry.path_length_m)
