from witwin.channel.core.edge_policy import EdgePolicy
from witwin.core import Scene


def test_edge_diffraction_is_explicitly_enabled_by_default():
    policy = EdgePolicy()

    assert policy.edge_diffraction is True
    assert policy.boundary_edge_policy == "half_plane"


def test_core_scene_does_not_own_mitsuba_loader_or_frequency():
    assert not hasattr(Scene, "load_mitsuba")
    assert "frequency" not in Scene.__dataclass_fields__
