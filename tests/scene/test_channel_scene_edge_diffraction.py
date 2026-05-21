from __future__ import annotations

import pytest

from witwin.channel import path
from witwin.channel.core.scene import EdgePolicy, Scene
from witwin.core import Material, Mesh, Structure


def _open_wall_scene(**scene_kwargs) -> Scene:
    mesh = Mesh(
        vertices=(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 3.0),
            (1.0, 0.0, 3.0),
        ),
        faces=((0, 1, 3), (0, 3, 2)),
        device="cpu",
    )
    return Scene(
        structures=[
            Structure(
                geometry=mesh,
                material=Material(),
                name="open_wall",
            )
        ],
        device="cpu",
        **scene_kwargs,
    )


def test_scene_constructor_rejects_solver_edge_policy_fields():
    with pytest.raises(TypeError):
        _open_wall_scene(edge_diffraction=False)
    with pytest.raises(TypeError):
        _open_wall_scene(edge_selection_mode="all_edges")
    with pytest.raises(TypeError):
        _open_wall_scene(boundary_edge_policy="exclude")


def test_solver_config_edge_policy_controls_boundary_edges():
    scene = _open_wall_scene()
    config = path.Config(
        edge_policy=EdgePolicy(
            edge_diffraction=True,
            boundary_edge_policy="half_plane",
            edge_selection_mode="all_edges",
        ),
    )
    edge_cache = scene.get_edge_data(1.5, edge_policy=config.edge_policy)

    assert scene.diffraction_edge_count(edge_policy=config.edge_policy) == 4
    assert edge_cache["edge_diffraction"] is True
    assert edge_cache["boundary_edge_policy"] == "half_plane"
    assert edge_cache["edge_data"]["n_edges"] == 4


def test_default_edge_policy_selects_all_boundary_edges():
    scene = _open_wall_scene()
    config = path.Config()
    edge_cache = scene.get_edge_data(1.5, edge_policy=config.edge_policy)

    assert config.edge_policy.vertical_only is False
    assert config.edge_policy.edge_selection_mode == "all_edges"
    assert scene.diffraction_edge_count(edge_policy=config.edge_policy) == 4
    assert edge_cache["edge_data"]["n_edges"] == 4


def test_get_edge_data_without_projection_skips_python_edge_views(monkeypatch):
    scene = _open_wall_scene()
    config = path.Config()

    def fail_projection(*args, **kwargs):
        raise AssertionError("diffraction point projection should be skipped")

    monkeypatch.setattr(Scene, "_make_diffraction_point", fail_projection)

    edge_cache = scene.get_edge_data(
        1.5,
        include_projection=False,
        edge_policy=config.edge_policy,
    )

    assert edge_cache["edge_data"]["n_edges"] == 4
    assert edge_cache["diffraction_points"] == ()


def test_solver_config_can_exclude_boundary_edges():
    scene = _open_wall_scene()
    config = path.Config(edge_policy=EdgePolicy(edge_diffraction=False))
    edge_cache = scene.get_edge_data(1.5, edge_policy=config.edge_policy)

    assert scene.diffraction_edge_count(edge_policy=config.edge_policy) == 0
    assert edge_cache["edge_diffraction"] is False
    assert edge_cache["boundary_edge_policy"] == "exclude"
    assert edge_cache["edge_data"] is None


def test_edge_policy_rejects_conflicting_boundary_edge_controls():
    with pytest.raises(ValueError, match="edge_diffraction=True.*boundary_edge_policy='exclude'"):
        EdgePolicy(edge_diffraction=True, boundary_edge_policy="exclude")

    with pytest.raises(ValueError, match="edge_diffraction=False.*boundary_edge_policy='half_plane'"):
        EdgePolicy(edge_diffraction=False, boundary_edge_policy="half_plane")
